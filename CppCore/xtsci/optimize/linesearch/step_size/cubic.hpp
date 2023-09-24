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

#include "xtensor-blas/xlinalg.hpp"

namespace xts {
namespace optimize {
namespace linesearch {
namespace step_size {
template <typename ScalarType>
class CubicStepSize : public StepSizeStrategy<ScalarType> {
public:
  ScalarType nextStep(ScalarType alpha_lo, ScalarType alpha_hi,
                      const ObjectiveFunction<ScalarType> &func,
                      const SearchState<ScalarType> &state) const override {
    // Fetch function values and derivatives at the endpoints
    ScalarType f_lo = func(state.x + alpha_lo);
    ScalarType f_hi = func(state.x + alpha_hi);
    auto g_lo_opt = func.gradient(state.x + alpha_lo);
    auto g_hi_opt = func.gradient(state.x + alpha_hi);
    // Ensure we have gradient values
    if (!g_lo_opt || !g_hi_opt) {
      // Handle the case where gradient is not available
      throw std::runtime_error("Gradient not available for given input.");
    }

    ScalarType g_lo = xt::linalg::dot(*g_lo_opt, state.direction)();
    ScalarType g_hi = xt::linalg::dot(*g_hi_opt, state.direction)();
    // Cubic interpolation coefficients
    ScalarType d1 = g_lo + g_hi - 3.0 * (f_lo - f_hi) / (alpha_lo - alpha_hi);
    ScalarType d2 =
        sign(alpha_hi - alpha_lo) * std::sqrt(d1 * d1 - g_lo * g_hi);
    ScalarType c3 = (g_hi + d2 - d1) / (alpha_hi - alpha_lo);
    ScalarType c2 = -(alpha_hi - alpha_lo) * (g_hi + 2.0 * d2 + d1) /
                    ((alpha_hi - alpha_lo) * (alpha_hi - alpha_lo));
    ScalarType c1 = (g_lo - d1 + d2) / (alpha_lo - alpha_hi);

    // Find minimum of the cubic polynomial in the interval by setting its
    // derivative to zero
    ScalarType t =
        -c1 / (3.0 * c3); // This is a simplification, assuming c3 is non-zero.

    // Ensure that the computed step is within the interval
    if (t < alpha_lo || t > alpha_hi) {
      // If outside of the interval, default to bisection
      return 0.5 * (alpha_lo + alpha_hi);
    }

    return t;
  }

private:
  // Sign function to handle potential negative values during the square root
  // computation
  static ScalarType sign(ScalarType x) {
    return (x >= ScalarType(0)) ? ScalarType(1) : ScalarType(-1);
  }
};
} // namespace step_size
} // namespace linesearch
} // namespace optimize
} // namespace xts
