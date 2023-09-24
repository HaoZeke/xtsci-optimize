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

namespace xts {
namespace optimize {
namespace linesearch {
namespace conditions {

template <typename ScalarType>
class CurvatureCondition : public LineSearchCondition<ScalarType> {
  ScalarType c_prime;

public:
  explicit CurvatureCondition(ScalarType c_prime_val = 0.9)
      : c_prime(c_prime_val) {}

  bool operator()(ScalarType alpha, const ObjectiveFunction<ScalarType> &func,
                  const xt::xarray<ScalarType> &x,
                  const xt::xarray<ScalarType> &direction) const override {
      auto lhs = xt::linalg::dot(*func.gradient(x + alpha * direction), direction)();
      auto rhs = c_prime * xt::linalg::dot(*func.gradient(x), direction)();
      return lhs >= rhs;
  };
};

} // namespace conditions
} // namespace linesearch
} // namespace optimize
} // namespace xts
