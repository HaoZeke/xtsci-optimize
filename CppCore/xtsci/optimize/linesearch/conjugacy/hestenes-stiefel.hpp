#pragma once
// MIT License
// Copyright 2023--present Rohit Goswami <HaoZeke>
#include <algorithm>
#include <functional>
#include <limits>
#include <string>
#include <vector>

#include "xtensor-blas/xlinalg.hpp"

#include "xtsci/optimize/linesearch/base.hpp"

namespace xts {
namespace optimize {
namespace linesearch {
namespace conjugacy {
template <typename ScalarType>
class HestenesStiefel : public ConjugacyCoefficientStrategy<ScalarType> {
public:
  ScalarType
  computeBeta(const ConjugacyContext<ScalarType> &ctx) const override {
    auto grad_change = ctx.current_gradient - ctx.previous_gradient;
    // [NJWS] Equation 5.46
    return (xt::linalg::dot(ctx.current_gradient, grad_change)() /
            xt::linalg::dot(grad_change, ctx.previous_direction)());
  }

  // References:
  // [NJWS] Nocedal, J., & Wright, S. (2006). Numerical optimization. Springer
};
} // namespace conjugacy
} // namespace linesearch
} // namespace optimize
} // namespace xts
