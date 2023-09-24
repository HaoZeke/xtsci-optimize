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

#include "xtsci/optimize/base.hpp"
#include "xtsci/optimize/linesearch/base.hpp"
#include "xtsci/optimize/linesearch/conditions/armijo.hpp"
#include "xtsci/optimize/linesearch/conditions/curvature.hpp"

namespace xts {
namespace optimize {
namespace linesearch {
namespace conditions {

template <typename ScalarType>
class GoldsteinUpperBoundCondition : public LineSearchCondition<ScalarType> {
  ScalarType c1;

public:
  explicit GoldsteinUpperBoundCondition(ScalarType c1_val = 0.0001)
      : c1(c1_val) {
    if (c1 <= 0 || c1 >= 0.5) {
      throw std::invalid_argument("c1 should be in the interval (0, 0.5).");
    }
  }

  bool operator()(ScalarType alpha, const ObjectiveFunction<ScalarType> &func,
                  const SearchState<ScalarType> &cstate) const override {
    auto [x, direction] = cstate;
    ScalarType lhs = func(x + alpha * direction);
    ScalarType f_at_x = func(x);
    ScalarType gradient_dot_dir =
        xt::linalg::dot(*func.gradient(x), direction)();

    ScalarType upper_bound = f_at_x + (1 - c1) * alpha * gradient_dot_dir;

    return lhs <= upper_bound;
  }
};

template <typename ScalarType>
class GoldsteinCondition : public LineSearchCondition<ScalarType> {
  ArmijoCondition<ScalarType> armijo;
  GoldsteinUpperBoundCondition<ScalarType> goldstein_upper;

public:
  explicit GoldsteinCondition(ScalarType c_armijo = 1e-4,
                              ScalarType c_upper = 1e-4)
      : armijo(c_armijo), goldstein_upper(c_upper) {}

  bool operator()(ScalarType alpha, const ObjectiveFunction<ScalarType> &func,
                  const SearchState<ScalarType> &cstate) const override {
    return armijo(alpha, func, cstate) && goldstein_upper(alpha, func, cstate);
  }
};

} // namespace conditions
} // namespace linesearch
} // namespace optimize
} // namespace xts
