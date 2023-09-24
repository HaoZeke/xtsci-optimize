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
namespace linesearch {
namespace conditions {

template <typename ScalarType>
class ArmijoCondition : public LineSearchCondition<ScalarType> {
  ScalarType c;

public:
  explicit ArmijoCondition(ScalarType c_val = 0.0001) : c(c_val) {}

  bool operator()(ScalarType alpha, const ObjectiveFunction<ScalarType> &func,
                  const xt::xarray<ScalarType> &x,
                  const xt::xarray<ScalarType> &direction) const override {
    ScalarType lhs = func(x + alpha * direction);
    ScalarType rhs =
        func(x) + c * alpha * xt::linalg::dot(*func.gradient(x), direction)();
    return lhs <= rhs;
  }
};

} // namespace conditions
} // namespace linesearch
} // namespace optimize
} // namespace xts
