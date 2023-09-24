#pragma once
// MIT License
// Copyright 2023--present Rohit Goswami <HaoZeke>
#include <fmt/ostream.h>

#include "xtensor/xarray.hpp"
#include "xtensor/xmath.hpp"
#include "xtensor/xnoalias.hpp"

#include "xtensor-blas/xlinalg.hpp"
#include "xtsci/optimize/linesearch/base.hpp"

namespace xts {
namespace optimize {
namespace minimize {

template <typename ScalarType>
class ADAMOptimizer : public linesearch::LineSearchOptimizer<ScalarType> {
public:
  ADAMOptimizer(linesearch::LineSearchStrategy<ScalarType> &strategy,
                ScalarType lr = 0.001, ScalarType beta1 = 0.9,
                ScalarType beta2 = 0.999, ScalarType epsilon = 1e-8)
      : linesearch::LineSearchOptimizer<ScalarType>(strategy), m_lr(lr),
        m_beta1(beta1), m_beta2(beta2), m_epsilon(epsilon) {}

  OptimizeResult<ScalarType>
  optimize(const ObjectiveFunction<ScalarType> &func,
           const SearchState<ScalarType> &initial_guess,
           const OptimizeControl<ScalarType> &control) const override {
    auto [x, _] = initial_guess;

    auto grad_opt = func.gradient(x);
    if (!grad_opt) {
      throw std::runtime_error("Gradient required for ADAM.");
    }

    auto gradient = *grad_opt;

    xt::xarray<ScalarType> m = xt::zeros_like(x);
    xt::xarray<ScalarType> v = xt::zeros_like(x);

    ScalarType beta1_pow = m_beta1;
    ScalarType beta2_pow = m_beta2;

    OptimizeResult<ScalarType> result;

    for (result.nit = 0; result.nit < control.max_iterations; ++result.nit) {
      if (control.verbose) {
        fmt::print("Iteration: {}\n", result.nit);
      }

      auto prev_gradient = gradient;
      auto new_grad_opt = func.gradient(x);
      if (!new_grad_opt) {
        throw std::runtime_error("Gradient required for ADAM.");
      }
      gradient = *new_grad_opt;

      // ADAM update rule
      m = m_beta1 * m + (1.0 - m_beta1) * gradient;
      v = m_beta2 * v + (1.0 - m_beta2) * xt::pow(gradient, 2.0);

      xt::xarray<ScalarType> m_hat = m / (1.0 - beta1_pow);
      xt::xarray<ScalarType> v_hat = v / (1.0 - beta2_pow);

      x -= m_lr * m_hat / (xt::sqrt(v_hat) + m_epsilon);

      beta1_pow *= m_beta1;
      beta2_pow *= m_beta2;

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
  ScalarType m_lr;    // Learning rate
  ScalarType m_beta1; // Exponential decay rate for the first moment estimates
  ScalarType m_beta2; // Exponential decay rate for the second moment estimates
  ScalarType m_epsilon; // A small constant for numerical stability
};

} // namespace minimize
} // namespace optimize
} // namespace xts
