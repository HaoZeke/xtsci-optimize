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
class MullerBrown : public ObjectiveFunction<ScalarType> {
  // TODO(rgoswami): Convert to references
  // From: https://arxiv.org/pdf/cond-mat/0108310.pdf
  // Also: https://arxiv.org/pdf/1701.01241.pdf
  // Domain is x∈[−1.5,1.2] and y∈[−0.2,2.0]
  // First minimum is at x = (-0.558, 1.442) with f(x) = -146.69948920058778
  // Second minimum is at x = (0.623, 0.028) with f(x) = -108.16665005353302
  // Third minimum is at x = (-0.050, 0.466) with f(x) = -80.76746772526472
  // First saddle point is at x = (0.212, 0.293) with f(x) = -72.24891965936473
  // Second saddle point is at x = (-0.822, 0.624) with f(x) =
  // -40.66484530104902
public:
  MullerBrown()
      : ObjectiveFunction<ScalarType>(
            /* minima */ {{-0.558, 1.442}, {0.623, 0.028}, {-0.050, 0.466}},
            /* saddles */ {{0.212, 0.293}, {-0.822, 0.624}}) {}

private:
  static constexpr std::array<ScalarType, 4> A = {-200, -100, -170, 15};
  static constexpr std::array<ScalarType, 4> a = {-1, -1, -6.5, 0.7};
  static constexpr std::array<ScalarType, 4> b = {0, 0, 11, 0.6};
  static constexpr std::array<ScalarType, 4> c = {-10, -10, -6.5, 0.7};
  static constexpr std::array<ScalarType, 4> x0 = {1, 0, -0.5, -1};
  static constexpr std::array<ScalarType, 4> y0 = {0, 0.5, 1.5, 1};

  ScalarType compute(const xt::xarray<ScalarType> &x) const override {
    ScalarType x_val = x(0);
    ScalarType y_val = x(1);
    ScalarType result = 0.0;

    for (size_t i = 0; i < 4; ++i) {
      result += A[i] * std::exp(a[i] * std::pow(x_val - x0[i], 2) +
                                b[i] * (x_val - x0[i]) * (y_val - y0[i]) +
                                c[i] * std::pow(y_val - y0[i], 2));
    }

    return result;
  }

  std::optional<xt::xarray<ScalarType>>
  compute_gradient(const xt::xarray<ScalarType> &x) const override {
    ScalarType x_val = x(0);
    ScalarType y_val = x(1);
    ScalarType df_dx = 0.0;
    ScalarType df_dy = 0.0;

    for (size_t i = 0; i < 4; ++i) {
      ScalarType exponent = a[i] * std::pow(x_val - x0[i], 2) +
                            b[i] * (x_val - x0[i]) * (y_val - y0[i]) +
                            c[i] * std::pow(y_val - y0[i], 2);
      ScalarType exp_value = std::exp(exponent);
      df_dx += A[i] * exp_value *
               (2 * a[i] * (x_val - x0[i]) + b[i] * (y_val - y0[i]));
      df_dy += A[i] * exp_value *
               (b[i] * (x_val - x0[i]) + 2 * c[i] * (y_val - y0[i]));
    }

    return xt::xarray<ScalarType>{df_dx, df_dy};
  }

  std::optional<xt::xarray<ScalarType>>
  compute_hessian(const xt::xarray<ScalarType> &x) const override {
    ScalarType x_val = x(0);
    ScalarType y_val = x(1);
    ScalarType d2f_dx2 = 0.0;
    ScalarType d2f_dy2 = 0.0;
    ScalarType d2f_dxdy = 0.0;

    for (size_t i = 0; i < 4; ++i) {
      ScalarType exponent = a[i] * std::pow(x_val - x0[i], 2) +
                            b[i] * (x_val - x0[i]) * (y_val - y0[i]) +
                            c[i] * std::pow(y_val - y0[i], 2);
      ScalarType exp_value = std::exp(exponent);

      d2f_dx2 += A[i] * exp_value *
                 ((2 * a[i]) * (2 * a[i]) + 2 * a[i] +
                  b[i] * b[i] * std::pow(y_val - y0[i], 2));
      d2f_dy2 += A[i] * exp_value *
                 ((2 * c[i]) * (2 * c[i]) + 2 * c[i] +
                  b[i] * b[i] * std::pow(x_val - x0[i], 2));
      d2f_dxdy += A[i] * exp_value *
                  (2 * a[i] * b[i] * (x_val - x0[i]) +
                   2 * c[i] * b[i] * (y_val - y0[i]) +
                   b[i] * b[i] * (x_val - x0[i]) * (y_val - y0[i]));
    }

    xt::xarray<ScalarType> hess = {{d2f_dx2, d2f_dxdy}, {d2f_dxdy, d2f_dy2}};
    return hess;
  }
};
} // namespace trial_functions
} // namespace optimize
} // namespace xts
