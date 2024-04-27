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
namespace restart {
class NJWSRestart : public RestartStrategy {
public:
  explicit NJWSRestart(ScalarType threshold = 0.5)
      : RestartStrategy(threshold) {}
  bool restart(const ConjugacyContext &ctx) const override {
    // [NJWS] Equation 5.52
    // Normalized cosine of the angle between the current and previous gradients
    auto deviation =
        std::abs(
            xt::linalg::dot(ctx.current_gradient, ctx.previous_gradient)()) /
        xt::linalg::dot(ctx.previous_gradient, ctx.previous_gradient)();
    return (deviation >= this->m_threshold);
  }

  // References:
  // [NJWS] Nocedal, J., & Wright, S. (2006). Numerical optimization. Springer
};

} // namespace restart
} // namespace nlcg
} // namespace optimize
} // namespace xts
