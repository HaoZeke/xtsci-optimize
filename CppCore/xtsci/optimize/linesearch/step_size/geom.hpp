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
class GeometricReductionStepSize : public StepSizeStrategy {
private:
  ScalarType beta;

public:
  explicit GeometricReductionStepSize(ScalarType b = 0.5) : beta(b) {}

  ScalarType nextStep(const AlphaState alpha, const Optimizable &,
                      const SearchState &) const override {
    return beta * alpha.init;
  }
};
} // namespace step_size
} // namespace linesearch
} // namespace optimize
} // namespace xts
