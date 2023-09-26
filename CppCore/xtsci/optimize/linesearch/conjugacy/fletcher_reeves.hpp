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
class FletcherReeves : public ConjugacyCoefficientStrategy<ScalarType> {
public:
  ScalarType
  computeBeta(const ConjugacyContext<ScalarType> &ctx) const override {
    // [NJWS] Equation 5.41a
    return xt::linalg::dot(ctx.current_gradient, ctx.current_gradient)() /
           xt::linalg::dot(ctx.previous_gradient, ctx.previous_gradient)();
  }

  // References:
  // [NJWS] Nocedal, J., & Wright, S. (2006). Numerical optimization. Springer
};

} // namespace conjugacy
} // namespace linesearch
} // namespace optimize
} // namespace xts
