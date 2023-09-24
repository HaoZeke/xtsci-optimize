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
class ConjugateGradientOptimizer
    : public linesearch::LineSearchOptimizer<ScalarType> {
public:
  ConjugateGradientOptimizer(
      linesearch::LineSearchStrategy<ScalarType> &strategy)
      : linesearch::LineSearchOptimizer<ScalarType>(strategy) {}

  OptimizeResult<ScalarType>
  optimize(const ObjectiveFunction<ScalarType> &func,
           const SearchState<ScalarType> &initial_guess,
           const OptimizeControl<ScalarType> &control) const override {
    auto [x, direction] = initial_guess;

    auto grad_opt = func.gradient(x);
    if (!grad_opt) {
      throw std::runtime_error(
          "Gradient required for conjugate gradient method.");
    }

    auto gradient = *grad_opt;
    direction = -gradient;

    if (control.verbose) {
      fmt::print("Initial x: {}\n", x);
      fmt::print("Initial gradient: {}\n", gradient);
    }

    OptimizeResult<ScalarType> result;

    for (result.nit = 0; result.nit < control.max_iterations; ++result.nit) {
      if (control.verbose) {
        fmt::print("Iteration: {}\n", result.nit);
      }
      ScalarType alpha = this->m_ls_strat.search(func, {x, direction});
      if (control.verbose) {
        fmt::print("Alpha: {}\n", alpha);
      }

      xt::noalias(x) += alpha * direction;
      if (control.verbose) {
        fmt::print("x: {}\n", x);
      }

      auto new_grad_opt = func.gradient(x);
      if (!new_grad_opt) {
        throw std::runtime_error(
            "Gradient required for conjugate gradient method.");
      }

      const auto &new_gradient = *new_grad_opt;
      if (control.verbose) {
        fmt::print("New gradient: {}\n", new_gradient);
      }

      if (xt::amax(xt::abs(new_gradient))() < control.tol) {
        break;
      }

      auto beta_expr = xt::linalg::dot(new_gradient, new_gradient) /
                       xt::linalg::dot(gradient, gradient);
      ScalarType beta = beta_expr();

      xt::noalias(direction) = -new_gradient + beta * direction;
      gradient = new_gradient; // Direct assignment (assumes ownership transfer
                               // if possible)

      if (control.verbose) {
        fmt::print("New direction: {}\n", direction);
      }
    }

    result.x = x;
    result.fun = func(x);
    result.jac = gradient;
    result.nfev = func.evaluation_counts().function_evals;
    result.njev = func.evaluation_counts().gradient_evals;
    result.nhev = func.evaluation_counts().hessian_evals;

    return result;
  }
};

} // namespace minimize
} // namespace optimize
} // namespace xts
