#include "xtsci/optimize/minimize/lbfgs.hpp"
#include "xtsci/optimize/numerics.hpp"

namespace xts::optimize::minimize {

void printOptimizationStep(size_t step, const ScalarType &energy,
                           const ScalarType &fmax) {
  // Get current time
  auto now = std::chrono::system_clock::now();
  auto now_c = std::chrono::system_clock::to_time_t(now);

  // Format the output
  fmt::print("LBFGS: {:3}   {:<8} {:16.9f} {:10.6f}\n", step,
             fmt::format("{:%H:%M:%S}", *std::localtime(&now_c)), energy, fmax);
}

void LBFGSOptimizer::step(const FObjFunc &func) {
  auto [c_x, _dir] = *m_cur;
  auto [c_grad, c_dir] = get_grad_dir(func, *m_cur);
  // Always try 1 first, but if it fails, search within a larger range
  ScalarType alpha =
      this->m_strat.get().search({1, 1e-6, 100}, func, {c_x, c_dir});
  auto s = alpha * c_dir;
  auto n_x = c_x + s;
  m_next = std::make_unique<SearchState>(n_x, *func.gradient(n_x));
  // TODO(rg): This is very confusing as written, actually y is just delta grad
  auto y = m_next->direction - c_grad;
  // Update the lists
  if (m_s_list.size() == m_corrections) {
    m_s_list.pop_front();
    m_y_list.pop_front();
    m_rho_list.pop_front();
  }
  m_s_list.push_back(s);
  m_y_list.push_back(y);
  m_rho_list.push_back(1.0 / xt::linalg::dot(y, s)());
  if (m_control.get().verbose) {
    auto energy = func(m_next->x);
    auto fmax = xt::linalg::norm(m_next->direction);
    printOptimizationStep(m_result.nit, energy, fmax);
  }
}

std::pair<ScalarVec, ScalarVec>
LBFGSOptimizer::get_grad_dir(const FObjFunc &func,
                             const SearchState &state) const {
  auto [x, _dir] = state;
  auto grad_opt = func.gradient(x);
  if (!grad_opt) {
    throw std::runtime_error("Gradient required for L-BFGS method.");
  }
  auto gradient = *grad_opt;
  auto direction = get_direction(gradient, m_s_list, m_y_list, m_rho_list);
  return std::make_pair(gradient, direction);
}

} // namespace xts::optimize::minimize
