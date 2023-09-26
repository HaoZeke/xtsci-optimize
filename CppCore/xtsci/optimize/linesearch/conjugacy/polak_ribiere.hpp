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
class PolakRibiere : public ConjugacyCoefficientStrategy<ScalarType> {
public:
  ScalarType
  computeBeta(const ConjugacyContext<ScalarType> &ctx) const override {
    return xt::linalg::dot(ctx.current_gradient,
                           ctx.current_gradient - ctx.previous_gradient)() /
           xt::linalg::dot(ctx.previous_gradient, ctx.previous_gradient)();
  }
};
} // namespace conjugacy
} // namespace linesearch
} // namespace optimize
} // namespace xts
