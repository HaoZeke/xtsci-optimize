#pragma once
// MIT License
// Copyright 2023--present Rohit Goswami <HaoZeke>
// clang-format off
#include <fmt/ostream.h>
#include <fmt/chrono.h>
#include <deque>
#include <vector>
// clang-format on

#include "xtsci/optimize/base.hpp"
#include "xtsci/optimize/numerics.hpp"

namespace xts {
namespace optimize {
namespace minimize {

void printOptimizationStep(size_t step, const ScalarType &energy,
                           const ScalarType &fmax);

class LBFGSOptimizer : public AbstractOptimizer {
private:
  size_t m_corrections; // Number of corrections to store
  // Two-deque to store s and y, which represent differences in x and
  // gradient, respectively
  std::deque<ScalarVec> m_s_list, m_y_list;
  std::deque<ScalarType> m_rho_list;

public:
  explicit LBFGSOptimizer(SearchStrategy &strategy,
                          size_t mem_list = 2 /* Typically 5 to 20 */)
      : AbstractOptimizer(strategy) {
    m_corrections = mem_list;
  }

protected:
  void step(const FObjFunc &func) override;

private:
  std::pair<ScalarVec, ScalarVec> get_grad_dir(const FObjFunc &func,
                                               const SearchState &state) const;
  ScalarVec get_direction(const ScalarVec &gradient,
                          const std::deque<ScalarVec> &s_list,
                          const std::deque<ScalarVec> &y_list,
                          const std::deque<ScalarType> &rho_list) const {
    std::vector<ScalarType> alpha_list(m_corrections, 0.0);
    auto q = gradient;

    // Two-loop recursion for L-BFGS
    for (int i = s_list.size() - 1; i >= 0; --i) {
      alpha_list[i] = rho_list[i] * xt::linalg::dot(s_list[i], q)();
      q -= alpha_list[i] * y_list[i];
    }

    if (!s_list.empty() && !y_list.empty()) {
      ScalarType scaling_factor =
          rho_list.back() *
          xt::linalg::dot(xt::xarray<ScalarType>(s_list.back()),
                          xt::xarray<ScalarType>(y_list.back()))();
      q *= scaling_factor;
    }

    xt::xarray<ScalarType> r = q;

    for (size_t i = 0; i < s_list.size(); ++i) {
      ScalarType beta = rho_list[i] * xt::linalg::dot(y_list[i], r)();
      r += s_list[i] * (alpha_list[i] - beta);
    }

    return -r;
  }
};
} // namespace minimize
} // namespace optimize
} // namespace xts
