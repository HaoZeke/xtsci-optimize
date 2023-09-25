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
class Himmelblau : public ObjectiveFunction<ScalarType> {
  // Domain is [-5, 5] x [-5, 5]
  // Global minima are at (3, 2) and (-2.805118, 3.131312) and (-3.779310,
  // -3.283186) and (3.584428, -1.848126)
private:
  ScalarType compute(const xt::xarray<ScalarType> &x) const override {
    ScalarType x_val = x(0);
    ScalarType y_val = x(1);
    return (x_val * x_val + y_val - 11) * (x_val * x_val + y_val - 11) +
           (x_val + y_val * y_val - 7) * (x_val + y_val * y_val - 7);
  }

  std::optional<xt::xarray<ScalarType>>
  compute_gradient(const xt::xarray<ScalarType> &x) const override {
    ScalarType x_val = x(0);
    ScalarType y_val = x(1);

    ScalarType df_dx = 4 * x_val * (x_val * x_val + y_val - 11) +
                       2 * (x_val + y_val * y_val - 7);
    ScalarType df_dy = 2 * (x_val * x_val + y_val - 11) +
                       4 * y_val * (x_val + y_val * y_val - 7);

    return xt::xarray<ScalarType>{df_dx, df_dy};
  }

  std::optional<xt::xarray<ScalarType>>
  compute_hessian(const xt::xarray<ScalarType> &x) const override {
    ScalarType x_val = x(0);
    ScalarType y_val = x(1);

    ScalarType d2f_dx2 = 4 * (3 * x_val * x_val + y_val - 11) + 2;
    ScalarType d2f_dxdy = 4 * x_val + 4 * y_val;
    ScalarType d2f_dydx = 4 * x_val + 4 * y_val;
    ScalarType d2f_dy2 = 4 * (x_val + 3 * y_val * y_val - 7) + 2;

    xt::xarray<ScalarType> hess = {{d2f_dx2, d2f_dxdy}, {d2f_dydx, d2f_dy2}};

    return hess;
  }
};

} // namespace trial_functions
} // namespace optimize
} // namespace xts
