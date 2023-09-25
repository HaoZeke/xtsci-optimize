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
class Eggholder : public ObjectiveFunction<ScalarType> {
  // Domain is -512 to 512
  // Minimum at (512, 404.2319) with value -959.6407
private:
  // Function evaluation
  ScalarType compute(const xt::xarray<ScalarType> &x) const override {
    ScalarType x_val = x(0);
    ScalarType y_val = x(1);
    return -(y_val + 47) *
               std::sin(std::sqrt(std::abs(x_val / 2 + (y_val + 47)))) -
           x_val * std::sin(std::sqrt(std::abs(x_val - (y_val + 47))));
  }

  // Gradient computation
  std::optional<xt::xarray<ScalarType>>
  compute_gradient(const xt::xarray<ScalarType> &x) const override {
    ScalarType x_val = x(0);
    ScalarType y_val = x(1);

    ScalarType df_dx =
        -std::sin(std::sqrt(std::abs(x_val / 2 + y_val + 47))) -
        (x_val * std::cos(std::sqrt(std::abs(x_val / 2 + y_val + 47))) /
         std::sqrt(std::abs(x_val / 2 + y_val + 47))) -
        std::sin(std::sqrt(std::abs(x_val - y_val - 47))) +
        (x_val * std::cos(std::sqrt(std::abs(x_val - y_val - 47))) /
         std::sqrt(std::abs(x_val - y_val - 47)));

    ScalarType df_dy =
        -std::cos(std::sqrt(std::abs(x_val / 2 + y_val + 47))) -
        ((y_val + 47) * std::cos(std::sqrt(std::abs(x_val / 2 + y_val + 47))) /
         std::sqrt(std::abs(x_val / 2 + y_val + 47))) +
        std::cos(std::sqrt(std::abs(x_val - y_val - 47))) -
        ((y_val + 47) * std::cos(std::sqrt(std::abs(x_val - y_val - 47))) /
         std::sqrt(std::abs(x_val - y_val - 47)));

    return xt::xarray<ScalarType>{df_dx, df_dy};
  }
};

} // namespace trial_functions
} // namespace optimize
} // namespace xts
