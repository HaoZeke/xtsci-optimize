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
class BisectionStepSize : public StepSizeStrategy {
public:
  ScalarType nextStep(const AlphaState alpha, const Optimizable &,
                      const SearchState &) const override {
    return (alpha.low + alpha.hi) / 2.0;
  }
};
} // namespace step_size
} // namespace linesearch
} // namespace optimize
} // namespace xts
