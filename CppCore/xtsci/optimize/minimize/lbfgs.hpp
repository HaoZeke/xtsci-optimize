#pragma once
// MIT License
// Copyright 2023--present Rohit Goswami <HaoZeke>
// clang-format off
#include <fmt/ostream.h>
#include <deque>
#include <vector>
// clang-format on

#include "xtensor/xnoalias.hpp"

#include "xtsci/optimize/linesearch/base.hpp"

#include "xtensor-blas/xlinalg.hpp"

namespace xts {
namespace optimize {
namespace minimize {

template <typename ScalarType>
class LBFGSOptimizer : public linesearch::LineSearchOptimizer<ScalarType> {
public:
  explicit LBFGSOptimizer(linesearch::LineSearchStrategy<ScalarType> &strategy,
                          size_t m)
      : linesearch::LineSearchOptimizer<ScalarType>(strategy), m(m) {}

  OptimizeResult<ScalarType>
  optimize(const ObjectiveFunction<ScalarType> &func,
           const SearchState<ScalarType> &initial_guess,
           const OptimizeControl<ScalarType> &control) const override {
    auto [x, dir_] = initial_guess;

    auto grad_opt = func.gradient(x);
    if (!grad_opt) {
      throw std::runtime_error("Gradient required for L-BFGS method.");
    }

    auto gradient = *grad_opt;

    if (control.verbose) {
      fmt::print("Initial x: {}\n", x);
      fmt::print("Initial gradient: {}\n", gradient);
    }

    OptimizeResult<ScalarType> result;

    // Two-deque to store s and y, which represent differences in x and
    // gradient, respectively
    std::deque<xt::xarray<ScalarType>> s_list, y_list;
    std::deque<ScalarType> rho_list;

    for (result.nit = 0; result.nit < control.max_iterations; ++result.nit) {
      if (control.verbose) {
        fmt::print("Iteration: {}\n", result.nit);
      }

      auto direction = get_direction(gradient, s_list, y_list, rho_list);

      ScalarType alpha = this->m_ls_strat.search(func, {x, direction});
      if (control.verbose) {
        fmt::print("Alpha: {}\n", alpha);
      }

      auto s = alpha * direction;
      xt::noalias(x) += s;

      if (control.verbose) {
        fmt::print("x: {}\n", x);
      }

      auto prev_gradient = gradient;
      auto new_grad_opt = func.gradient(x);
      if (!new_grad_opt) {
        throw std::runtime_error("Gradient required for L-BFGS method.");
      }

      gradient = *new_grad_opt;
      auto y = gradient - prev_gradient;

      // Update the lists
      if (s_list.size() == m) {
        s_list.pop_front();
        y_list.pop_front();
        rho_list.pop_front();
      }
      s_list.push_back(s);
      y_list.push_back(y);
      rho_list.push_back(1.0 / xt::linalg::dot(y, s)());

      if (control.verbose) {
        fmt::print("New gradient: {}\n", gradient);
      }

      if (xt::amax(xt::abs(gradient))() < control.tol) {
        break;
      }
    }

    result.x = x;
    result.fun = func(x);
    result.jac = gradient;

    return result;
  }

private:
  size_t m; // Number of corrections to store

  xt::xarray<ScalarType>
  get_direction(const xt::xarray<ScalarType> &gradient,
                const std::deque<xt::xarray<ScalarType>> &s_list,
                const std::deque<xt::xarray<ScalarType>> &y_list,
                const std::deque<ScalarType> &rho_list) const {
    std::vector<ScalarType> alpha_list(m, 0.0);
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
