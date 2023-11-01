#pragma once
// MIT License
// Copyright 2023--present Rohit Goswami <HaoZeke>
#include <algorithm>
#include <functional>
#include <limits>
#include <string>
#include <vector>

#include "xtsci/optimize/base.hpp"
#include "xtsci/optimize/linesearch/base.hpp"

namespace xts {
namespace optimize {
namespace linesearch {
namespace step_size {

template <typename ScalarType>
class CubicInterpolationStepSize : public StepSizeStrategy<ScalarType> {
public:
  ScalarType nextStep(const AlphaState<ScalarType> alpha,
                      const func::ObjectiveFunction<ScalarType> &func,
                      const SearchState<ScalarType> &cstate) const override {
    fmt::printf("Warning: CubicInterpolationStepSize is often unstable and is "
                "only provided for reference"
                "Use HermiteInterpolationStepSize instead.\n");
    ScalarType fa = func(cstate.x + alpha.low * cstate.direction);
    ScalarType fb = func(cstate.x + alpha.hi * cstate.direction);

    ScalarType fpa = func.directional_derivative(
        cstate.x + alpha.low * cstate.direction, cstate.direction);
    ScalarType fpb = func.directional_derivative(
        cstate.x + alpha.hi * cstate.direction, cstate.direction);

    ScalarType z = 3.0 * (fa - fb) / (alpha.hi - alpha.low) + fpa + fpb;
    ScalarType w = std::sqrt(std::max(
        static_cast<ScalarType>(0),
        z * z - fpa * fpb)); // Avoid negative values under the square root
    ScalarType m = (fpb + w - z) / (fpb - fpa + 2.0 * w);

    ScalarType step = alpha.hi - m * (alpha.hi - alpha.low);

    // If the cubic interpolation value is outside of the interval [low, hi],
    // revert to bisection
    if (step < alpha.low || step > alpha.hi) {
      return (alpha.low + alpha.hi) / 2.0;
    }

    return step;
  }
};
} // namespace step_size
} // namespace linesearch
} // namespace optimize
} // namespace xts
