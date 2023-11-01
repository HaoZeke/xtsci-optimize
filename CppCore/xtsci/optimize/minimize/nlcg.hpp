#pragma once
// MIT License
// Copyright 2023--present Rohit Goswami <HaoZeke>
#include <fmt/ostream.h>

#include "xtensor/xbuilder.hpp"
#include "xtensor/xnoalias.hpp"

#include "xtsci/optimize/linesearch/base.hpp"
#include "xtsci/optimize/linesearch/conditions/armijo.hpp"
#include "xtsci/optimize/nlcg/base.hpp"

#include "xtensor-blas/xlinalg.hpp"

namespace xts {
namespace optimize {
namespace minimize {

template <typename ScalarType>
class ConjugateGradientOptimizer
    : public linesearch::LineSearchOptimizer<ScalarType> {
public:
  std::reference_wrapper<nlcg::ConjugacyCoefficientStrategy<ScalarType>> m_conj;
  std::reference_wrapper<nlcg::RestartStrategy<ScalarType>> m_restart;
  ConjugateGradientOptimizer(
      linesearch::LineSearchStrategy<ScalarType> &strategy,
      nlcg::ConjugacyCoefficientStrategy<ScalarType> &conjugacy_strategy,
      nlcg::RestartStrategy<ScalarType> &restart_strategy)
      : linesearch::LineSearchOptimizer<ScalarType>(strategy),
        m_conj(conjugacy_strategy), m_restart(restart_strategy) {}

  OptimizeResult<ScalarType>
  optimize(const func::ObjectiveFunction<ScalarType> &func,
           const SearchState<ScalarType> &initial_guess,
           const OptimizeControl<ScalarType> &control) const override {
    auto [x, direction] = initial_guess;

    auto grad_opt = func.gradient(x);
    if (!grad_opt) {
      throw std::runtime_error(
          "Gradient required for conjugate gradient method.");
    }

    auto gradient = *grad_opt;
    xt::xarray<ScalarType> old_gradient = gradient;
    direction = -gradient;

    if (control.verbose) {
      fmt::print("Initial x: {}\n", x);
      fmt::print("Initial gradient: {}\n", gradient);
    }

    OptimizeResult<ScalarType> result;
    nlcg::ConjugacyContext<ScalarType> conj_ctx;

    ScalarType alpha = 1.0;
    // [NJWS] Algorithm 5.4
    for (result.nit = 0; result.nit < control.max_iterations; ++result.nit) {
      if (control.verbose) {
        fmt::print("Iteration: {}\n", result.nit);
      }

      // 1. Line search to get alpha for the current direction.
      // [NJWS] Equation 5.43a
      alpha = this->m_ls_strat.search({1.0, 1e-6, 10}, func, {x, direction});
      if (control.verbose) {
        fmt::print("Alpha: {}\n", alpha);
      }

      // 2. Update x using the current direction and alpha.
      auto proposed_move = alpha * direction;
      ScalarType proposed_move_norm = xt::linalg::norm(proposed_move);

      // TODO(rg): Document this non-standard behavior
      // If the proposed move is larger than maxmove, then scale the move down
      ScalarType scale_factor = 1.0;
      if (proposed_move_norm > control.maxmove) {
        scale_factor = control.maxmove / proposed_move_norm;
      }

      xt::noalias(x) += scale_factor * proposed_move;

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

      conj_ctx.current_gradient = gradient;
      conj_ctx.previous_gradient = old_gradient;
      conj_ctx.previous_direction = direction;

      // 4. Compute the beta coefficient.
      ScalarType beta = m_conj.get().computeBeta(conj_ctx);
      if (m_restart.get().restart(conj_ctx)) {
        fmt::print("Restarting due to the restart strategy\n");
        beta = 0;
      }

      if (control.verbose) {
        fmt::print("Beta: {}\n", beta);
      }

      // 5. Update the direction.
      direction = -gradient + beta * direction;

      if (control.verbose) {
        fmt::print("New direction: {}\n", direction);
      }
      old_gradient = gradient;
    }

    result.x = x;
    result.fun = func(x);
    result.jac = gradient;
    result.nfev = func.evaluation_counts().function_evals;
    result.njev = func.evaluation_counts().gradient_evals;
    result.nhev = func.evaluation_counts().hessian_evals;
    result.nufg = func.evaluation_counts().unique_func_grad;

    return result;
  }

  // References:
  // [NJWS] Nocedal, J., & Wright, S. (2006). Numerical optimization. Springer
};

} // namespace minimize
} // namespace optimize
} // namespace xts
