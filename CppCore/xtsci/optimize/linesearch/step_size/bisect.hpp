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
class BisectionStep : public StepSizeStrategy<ScalarType> {
public:
  ScalarType nextStep(ScalarType alpha_lo, ScalarType alpha_hi) const override {
    return 0.5 * (alpha_lo + alpha_hi);
  }
};
} // namespace step_size
} // namespace linesearch
} // namespace optimize
} // namespace xts
