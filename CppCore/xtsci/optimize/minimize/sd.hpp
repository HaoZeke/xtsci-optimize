#pragma once
// MIT License
// Copyright 2023--present Rohit Goswami <HaoZeke>
#include <fmt/ostream.h>

#include "xtensor/xnoalias.hpp"

#include "xtsci/optimize/linesearch/base.hpp"
#include "xtsci/optimize/linesearch/conditions/armijo.hpp"
#include "xtsci/optimize/nlcg/base.hpp"

#include "xtensor-blas/xlinalg.hpp"

namespace xts {
namespace optimize {
namespace minimize {

template <typename ScalarType>
class SteepestDescentOptimizer
    : public linesearch::LineSearchOptimizer<ScalarType> {
public:
  explicit SteepestDescentOptimizer(
      linesearch::LineSearchStrategy<ScalarType> &strategy)
      : linesearch::LineSearchOptimizer<ScalarType>(strategy) {}

  OptimizeResult<ScalarType>
  optimize(const func::ObjectiveFunction<ScalarType> &func,
           const SearchState<ScalarType> &initial_guess,
           const OptimizeControl<ScalarType> &control) const override {
    auto [x, direction] = initial_guess;

    auto grad_opt = func.gradient(x);
    if (!grad_opt) {
      throw std::runtime_error(
          "Gradient required for steepest descent method.");
    }

    auto gradient = *grad_opt;
    xt::xarray<ScalarType> old_gradient = gradient;
    direction = -gradient;

    if (control.verbose) {
      fmt::print("Initial x: {}\n", x);
      fmt::print("Initial gradient: {}\n", gradient);
    }

    OptimizeResult<ScalarType> result;

    ScalarType alpha = 1.0;
    for (result.nit = 0; result.nit < control.max_iterations; ++result.nit) {
      if (control.verbose) {
        fmt::print("Iteration: {}\n", result.nit);
      }

      // 1. Line search to get alpha for the current direction.
      // [NJWS] Equation 5.43a
      alpha = this->m_ls_strat.search({1.0, 1e-6, control.maxmove}, func,
                                      {x, direction});
      if (alpha == 0) {
        if (control.verbose) {
          fmt::print("Line search failed.\n");
        }
        break;
      }
      if (control.verbose) {
        fmt::print("Alpha: {}\n", alpha);
      }

      // 2. Update x using the current direction and alpha.
      // Steepest descent, so it is always the negative of the gradient
      xt::noalias(x) += alpha * direction;
      if (control.verbose) {
        fmt::print("x: {}\n", x);
      }

      // 3. Compute the new gradient at the updated x.
      auto new_grad_opt = func.gradient(x);
      gradient = *new_grad_opt; // Updating the gradient
      if (control.verbose) {
        fmt::print("New gradient: {}\n", gradient);
      }

      if (xt::amax(xt::abs(gradient))() < control.gtol) {
        if (control.verbose) {
          fmt::print("Change in gradient below threshold.\n");
        }
        break;
      }

      // 5. Update the direction.
      direction = -gradient;

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

  // References:
  // [NJWS] Nocedal, J., & Wright, S. (2006). Numerical optimization. Springer
};

} // namespace minimize
} // namespace optimize
} // namespace xts
