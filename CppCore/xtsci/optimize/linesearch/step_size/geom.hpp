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
class GeometricReductionStepSize : public StepSizeStrategy<ScalarType> {
private:
  ScalarType beta;

public:
  explicit GeometricReductionStepSize(ScalarType b = 0.5) : beta(b) {}

  ScalarType nextStep(const AlphaState<ScalarType> alpha,
                      const ObjectiveFunction<ScalarType> &,
                      const SearchState<ScalarType> &) const override {
    return beta * alpha.init;
  }
};
} // namespace step_size
} // namespace linesearch
} // namespace optimize
} // namespace xts
