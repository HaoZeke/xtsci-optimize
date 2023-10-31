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
class SecantStepSize : public StepSizeStrategy<ScalarType> {
public:
  ScalarType nextStep(const AlphaState<ScalarType> alpha,
                      const func::ObjectiveFunction<ScalarType> &func,
                      const SearchState<ScalarType> &cstate) const override {
    ScalarType fa = func(cstate.x + alpha.low * cstate.direction);
    ScalarType fb = func(cstate.x + alpha.hi * cstate.direction);
    ScalarType fpa = func.directional_derivative(
        cstate.x + alpha.low * cstate.direction, cstate.direction);
    ScalarType fpb = func.directional_derivative(
        cstate.x + alpha.hi * cstate.direction, cstate.direction);
    // Secant method formula
    ScalarType step = alpha.hi - fpb * (alpha.hi - alpha.low) / (fpb - fpa);
    // If the secant value is outside of the interval [low, hi], revert to
    // bisection
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
