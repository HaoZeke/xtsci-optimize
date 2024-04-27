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
class FletcherReeves : public ConjugacyCoefficientStrategy {
public:
  ScalarType computeBeta(const ConjugacyContext &ctx) const override {
    // [NJWS] Equation 5.41a
    return xt::linalg::dot(ctx.current_gradient, ctx.current_gradient)() /
           xt::linalg::dot(ctx.previous_gradient, ctx.previous_gradient)();
  }

  // References:
  // [NJWS] Nocedal, J., & Wright, S. (2006). Numerical optimization. Springer
};

} // namespace conjugacy
} // namespace nlcg
} // namespace optimize
} // namespace xts
