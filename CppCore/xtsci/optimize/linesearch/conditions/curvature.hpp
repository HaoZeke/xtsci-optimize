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

namespace xts {
namespace optimize {
namespace linesearch {
namespace conditions {

class CurvatureCondition : public SearchCondition {
public:
  ScalarType c_prime;
  explicit CurvatureCondition(ScalarType c_prime_val = 0.9)
      : c_prime(c_prime_val) {}
  bool operator()(ScalarType alpha, const Optimizable &optobj,
                  const SearchState &cstate) const override {
    auto [x, direction] = cstate;
    auto lhs =
        xt::linalg::dot(*optobj.gradient(x + alpha * direction), direction)();
    auto rhs = c_prime * xt::linalg::dot(*optobj.gradient(x), direction)();
    return lhs >= rhs;
  };
};

class StrongCurvatureCondition : public SearchCondition {
public:
  ScalarType c;
  explicit StrongCurvatureCondition(ScalarType c_val = 0.9) : c(c_val) {}

  bool operator()(ScalarType alpha, const Optimizable &optobj,
                  const SearchState &cstate) const override {
    auto [x, direction] = cstate;
    auto grad_phi_alpha =
        xt::linalg::dot(*optobj.gradient(x + alpha * direction), direction)();
    auto grad_phi_0 = optobj.directional_derivative(x, direction);
    return std::abs(grad_phi_alpha) <= c * std::abs(grad_phi_0);
  }
};

} // namespace conditions
} // namespace linesearch
} // namespace optimize
} // namespace xts
