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
public:
  static constexpr ScalarType phi = (1 + std::sqrt(5.0)) / 2.0;
  ScalarType nextStep(const AlphaState<ScalarType> alpha,
                      const func::ObjectiveFunction<ScalarType> &,
                      const SearchState<ScalarType> &) const override {
    ScalarType range = alpha.hi - alpha.low;
    ScalarType step = range / phi;

    // Choose point closer to alpha.init
    if (std::abs(alpha.init - alpha.low) < std::abs(alpha.init - alpha.hi)) {
      return alpha.low + step;
    } else {
      return alpha.hi - step;
    }
  }
};
} // namespace step_size
} // namespace linesearch
} // namespace optimize
} // namespace xts
