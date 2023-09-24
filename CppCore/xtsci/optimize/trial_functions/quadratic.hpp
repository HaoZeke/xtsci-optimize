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

template <typename ScalarType>
class QuadraticFunction : public xts::optimize::ObjectiveFunction<ScalarType> {
public:
  ScalarType operator()(const xt::xarray<ScalarType> &x) const override {
    return xt::linalg::dot(x, x)(0); // x^T x
  }

  std::optional<xt::xarray<ScalarType>>
  gradient(const xt::xarray<ScalarType> &x) const override {
    return 2.0 * x; // 2x
  }

  std::optional<xt::xarray<ScalarType>>
  hessian(const xt::xarray<ScalarType> &x) const override {
    // For a quadratic function, the Hessian is constant: 2I where I is the
    // identity matrix.
    return 2.0 * xt::eye(x.size());
  }
};

} // namespace trial_functions
} // namespace optimize
} // namespace xts
