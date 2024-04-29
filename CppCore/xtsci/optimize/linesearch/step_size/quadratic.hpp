#pragma once
// MIT License
// Copyright 2023--present Rohit Goswami <HaoZeke>
#include <algorithm>
#include <functional>
#include <limits>
#include <string>
#include <vector>

#include "xtsci/optimize/base.hpp"

namespace xts {
namespace optimize {
namespace linesearch {
namespace step_size {

class QuadraticInterpolationStepSize : public StepSizeStrategy {
public:
  QuadraticInterpolationStepSize() {}

  ScalarType nextStep(const AlphaState alpha, const Optimizable &optobj,
                      const SearchState &cstate) const override {
    auto phi = [&](ScalarType a_val) {
      return optobj(cstate.x + a_val * cstate.direction);
    };

    ScalarType phi_low = phi(alpha.low);
    ScalarType phi_hi = phi(alpha.hi);
    ScalarType phi_init = phi(alpha.init);

    ScalarType denominator = (phi_hi - phi_init) * alpha.low +
                             (phi_init - phi_low) * alpha.hi +
                             (phi_low - phi_hi) * alpha.init;
    ScalarType epsilon = 1e-10; // small regularization term

    if (std::abs(denominator) < epsilon) {
      return (alpha.low + alpha.hi) /
             2.0; // Fallback to bisection if denominator is too close to zero
    }

    ScalarType numerator = alpha.low * alpha.low * (phi_hi - phi_init) +
                           alpha.hi * alpha.hi * (phi_init - phi_low) +
                           alpha.init * alpha.init * (phi_low - phi_hi);

    ScalarType interpolated_value = numerator / (2.0 * denominator);

    if (interpolated_value == 0.0 || std::isnan(interpolated_value) ||
        std::isinf(interpolated_value)) {
      fmt::print("Warning: Interpolated value is zero, NaN or Inf. Falling "
                 "back to bisection.");
      return (alpha.low + alpha.hi) / 2.0; // Additional fallback
    }

    return interpolated_value;
  }
};
} // namespace step_size
} // namespace linesearch
} // namespace optimize
} // namespace xts
