#pragma once
// MIT License
// Copyright 2023--present Rohit Goswami <HaoZeke>
#include <fmt/ostream.h>

#include "xtensor/xnoalias.hpp"

#include "xtsci/optimize/linesearch/base.hpp"

#include "xtensor-blas/xlinalg.hpp"

namespace xts {
namespace optimize {
namespace minimize {

template <typename ScalarType>
class BFGSOptimizer : public linesearch::LineSearchOptimizer<ScalarType> {
public:
  explicit BFGSOptimizer(linesearch::LineSearchStrategy<ScalarType> &strategy)
      : linesearch::LineSearchOptimizer<ScalarType>(strategy) {}

  OptimizeResult<ScalarType>
  optimize(const ObjectiveFunction<ScalarType> &func,
           const SearchState<ScalarType> &initial_guess,
           const OptimizeControl<ScalarType> &control) const override {
    auto [x, dir_] =
        initial_guess; // Note: BFGS doesn't need an initial direction

    auto grad_opt = func.gradient(x);
    if (!grad_opt) {
      throw std::runtime_error("Gradient required for BFGS method.");
    }

    auto gradient = *grad_opt;
    xt::xarray<ScalarType> B_inv = xt::eye({x.size(), x.size()}, 0);

    if (control.verbose) {
      fmt::print("Initial x: {}\n", x);
      fmt::print("Initial gradient: {}\n", gradient);
    }

    OptimizeResult<ScalarType> result;

    for (result.nit = 0; result.nit < control.max_iterations; ++result.nit) {
      if (control.verbose) {
        fmt::print("Iteration: {}\n", result.nit);
      }

      auto direction = -xt::linalg::dot(B_inv, gradient);
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
        throw std::runtime_error("Gradient required for BFGS method.");
      }

      gradient = *new_grad_opt;
      auto y = gradient - prev_gradient;

      auto rho = 1.0 / xt::linalg::dot(y, s)();

      auto term1 =
          xt::eye({x.size(), x.size()}, 0) - rho * xt::linalg::outer(s, y);
      auto term2 =
          xt::eye({x.size(), x.size()}, 0) - rho * xt::linalg::outer(y, s);
      B_inv = xt::linalg::dot(term1, xt::linalg::dot(B_inv, term2)) +
              rho * xt::linalg::outer(s, s);

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
};

} // namespace minimize
} // namespace optimize
} // namespace xts
