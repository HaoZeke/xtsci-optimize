#pragma once
// MIT License
// Copyright 2023--present Rohit Goswami <HaoZeke>
#include <algorithm>
#include <functional>
#include <limits>
#include <string>
#include <vector>

#include "xtensor-blas/xlinalg.hpp"

#include "xtsci/optimize/nlcg/base.hpp"

namespace xts {
namespace optimize {
namespace nlcg {
namespace conjugacy {
class PolakRibiere : public ConjugacyCoefficientStrategy {
public:
  ScalarType computeBeta(const ConjugacyContext &ctx) const override {
    auto grad_change = ctx.current_gradient - ctx.previous_gradient;
    // [NJWS] Equation 5.44
    return xt::linalg::dot(ctx.current_gradient, grad_change)() /
           xt::linalg::dot(ctx.previous_gradient, ctx.previous_gradient)();
  }

  // References:
  // [NJWS] Nocedal, J., & Wright, S. (2006). Numerical optimization. Springer
};
} // namespace conjugacy
} // namespace nlcg
} // namespace optimize
} // namespace xts
