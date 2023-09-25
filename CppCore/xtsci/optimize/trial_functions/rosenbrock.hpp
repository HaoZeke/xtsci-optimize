#pragma once
// MIT License
// Copyright 2023--present Rohit Goswami <HaoZeke>
#include <algorithm>
#include <functional>
#include <limits>
#include <optional>
#include <string>
#include <vector>

#include "xtensor-blas/xlinalg.hpp"

#include "xtsci/optimize/linesearch/base.hpp"

namespace xts {
namespace optimize {
namespace trial_functions {

template <typename ScalarType = double>
class Rosenbrock : public ObjectiveFunction<ScalarType> {
  // Domain is R^2
  // Global minimum is at x = (1, 1) with f(x) = 0
private:
  // Function evaluation
  ScalarType compute(const xt::xarray<ScalarType> &x) const override {
    ScalarType x_val = x(0);
    ScalarType y_val = x(1);
    return (1 - x_val) * (1 - x_val) +
           100 * (y_val - x_val * x_val) * (y_val - x_val * x_val);
  }

  // Gradient computation
  std::optional<xt::xarray<ScalarType>>
  compute_gradient(const xt::xarray<ScalarType> &x) const override {
    ScalarType x_val = x(0);
    ScalarType y_val = x(1);

    ScalarType df_dx = -2 * (1 - x_val) - 400 * x_val * (y_val - x_val * x_val);
    ScalarType df_dy = 200 * (y_val - x_val * x_val);

    return xt::xarray<ScalarType>{df_dx, df_dy};
  }

  // Hessian computation
  std::optional<xt::xarray<ScalarType>>
  compute_hessian(const xt::xarray<ScalarType> &x) const override {
    ScalarType x_val = x(0);
    ScalarType y_val = x(1);

    ScalarType d2f_dx2 = 2 - 400 * y_val + 1200 * x_val * x_val;
    ScalarType d2f_dxdy = -400 * x_val;
    ScalarType d2f_dydx = -400 * x_val;
    ScalarType d2f_dy2 = 200;

    xt::xarray<ScalarType> hess = {{d2f_dx2, d2f_dxdy}, {d2f_dydx, d2f_dy2}};

    return hess;
  }
};

} // namespace trial_functions
} // namespace optimize
} // namespace xts
