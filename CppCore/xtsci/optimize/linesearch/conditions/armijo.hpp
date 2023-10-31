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
public:
  ScalarType c;
  explicit ArmijoCondition(ScalarType c_val = 0.0001) : c(c_val) {}

  bool operator()(ScalarType alpha,
                  const func::ObjectiveFunction<ScalarType> &func,
                  const SearchState<ScalarType> &cstate) const override {
    auto [x, direction] = cstate;
    ScalarType lhs = func(x + alpha * direction);
    ScalarType rhs =
        func(x) + c * alpha * func.directional_derivative(x, direction);
    return lhs <= rhs;
  }
};

} // namespace conditions
} // namespace linesearch
} // namespace optimize
} // namespace xts
