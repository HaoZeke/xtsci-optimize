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
// This will fit a cubic Hermite polynomial to the function and its derivative
template <typename ScalarType>
class HermiteInterpolationStepSize : public StepSizeStrategy<ScalarType> {
public:
  ScalarType nextStep(const AlphaState<ScalarType> alpha,
                      const func::ObjectiveFunction<ScalarType> &func,
                      const SearchState<ScalarType> &cstate) const override {
    ScalarType x0 = alpha.low;
    ScalarType x1 = alpha.hi;
    ScalarType f0 = func(cstate.x + x0 * cstate.direction);
    ScalarType f1 = func(cstate.x + x1 * cstate.direction);
    ScalarType df0 = func.directional_derivative(
        cstate.x + x0 * cstate.direction, cstate.direction);
    ScalarType df1 = func.directional_derivative(
        cstate.x + x1 * cstate.direction, cstate.direction);

    // Compute coefficients for the cubic Hermite polynomial
    ScalarType d = f0;
    ScalarType c = df0;
    ScalarType b = 3 * (f1 - f0) - 2 * df0 - df1;
    ScalarType a = df0 + df1 - 2 * (f1 - f0);

    // Derive the polynomial: ax^3 + bx^2 + cx + d => 3ax^2 + 2bx + c
    // Solve for x using the quadratic formula.
    ScalarType discriminant = b * b - 4 * a * c;

    if (discriminant < 0) {
      // No real roots, just return the midpoint
      return (x0 + x1) / 2.0;
    }

    ScalarType root1 = (-b + std::sqrt(discriminant)) / (2 * a);
    ScalarType root2 = (-b - std::sqrt(discriminant)) / (2 * a);

    // Choose the root that lies in the interval [x0, x1].
    // Also, to ensure it's a minimum, the second derivative (which is 6ax + 2b)
    // should be positive.
    if (x0 <= root1 && root1 <= x1 && (6 * a * root1 + 2 * b) > 0) {
      return root1;
    } else if (x0 <= root2 && root2 <= x1 && (6 * a * root2 + 2 * b) > 0) {
      return root2;
    } else {
      // If neither root is suitable, just return the midpoint.
      return (x0 + x1) / 2.0;
    }
  }
};

} // namespace step_size
} // namespace linesearch
} // namespace optimize
} // namespace xts
