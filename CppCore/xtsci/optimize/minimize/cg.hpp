#pragma once
// MIT License
// Copyright 2023--present Rohit Goswami <HaoZeke>
#include <fmt/ostream.h>

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

    if (!func.gradient(x)) {
      throw std::runtime_error(
          "Gradient required for conjugate gradient method.");
    }

    xt::xarray<ScalarType> gradient = func.gradient(x).value();
    // XXX: Don't do this if the user provides a gradient
    direction = -gradient;
    fmt::print("Initial direction: {}\n", direction);
    fmt::print("Initial x: {}\n", x);

    OptimizeResult<ScalarType> result;

    for (result.nit = 0; result.nit < control.max_iterations; ++result.nit) {
      fmt::print("Iteration: {}\n", result.nit);
      ScalarType alpha = this->m_ls_strat.search(func, {x, direction});
      // alpha = 1;
      fmt::print("Alpha: {}\n", alpha);

      x += alpha * direction;
      fmt::print("x: {}\n", x);

      if (!func.gradient(x)) {
        throw std::runtime_error(
            "Gradient required for conjugate gradient method.");
      }
      xt::xarray<ScalarType> new_gradient = func.gradient(x).value();
      fmt::print("New gradient: {}\n", new_gradient);

      if (xt::amax(xt::abs(new_gradient))() < control.tol) {
        break;
      }

      ScalarType beta = xt::linalg::dot(new_gradient, new_gradient)() /
                        xt::linalg::dot(gradient, gradient)();
      direction = -new_gradient + beta * direction;
      gradient = new_gradient;
      fmt::print("New direction: {}\n", direction);
      // if (result.nit > 5) {
      //   exit(1);
      // }
    }

    result.x = x;
    result.fun = func(x);
    if (func.gradient(x)) {
      result.jac = func.gradient(x).value();
    }

    return result;
  }
};
} // namespace minimize

} // namespace optimize

} // namespace xts
