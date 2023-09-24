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
class GoldenStepSize : public StepSizeStrategy<ScalarType> {
private:
  ScalarType m_tolerance;
  ScalarType m_golden_ratio;

public:
  explicit GoldenStepSize(ScalarType tolerance = 1e-5)
      : m_tolerance(tolerance), m_golden_ratio((std::sqrt(5.0) - 1.0) / 2.0) {}

  ScalarType nextStep(ScalarType alpha_lo, ScalarType alpha_hi,
                      const ObjectiveFunction<ScalarType> &func,
                      const SearchState<ScalarType> &state) const override {
    ScalarType alpha1 =
        alpha_lo + (1.0 - m_golden_ratio) * (alpha_hi - alpha_lo);
    ScalarType alpha2 = alpha_lo + m_golden_ratio * (alpha_hi - alpha_lo);

    while (alpha_hi - alpha_lo > m_tolerance) {
      if (func(state.x + alpha1 * state.direction) <
          func(state.x + alpha2 * state.direction)) {
        alpha_hi = alpha2;
        alpha2 = alpha1;
        alpha1 = alpha_lo + (1.0 - m_golden_ratio) * (alpha_hi - alpha_lo);
      } else {
        alpha_lo = alpha1;
        alpha1 = alpha2;
        alpha2 = alpha_lo + m_golden_ratio * (alpha_hi - alpha_lo);
      }
    }

    return (alpha_lo + alpha_hi) / 2.0;
  }
};
} // namespace step_size
} // namespace linesearch
} // namespace optimize
} // namespace xts
