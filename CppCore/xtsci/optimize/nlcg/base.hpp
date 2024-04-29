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
namespace nlcg {
struct ConjugacyContext {
  ScalarVec current_gradient;
  ScalarVec previous_gradient;
  ScalarVec previous_direction;
};

class ConjugacyCoefficientStrategy {
public:
  virtual ~ConjugacyCoefficientStrategy() = default;

  virtual ScalarType computeBeta(const ConjugacyContext &context) const = 0;
};

class RestartStrategy {
public:
  ScalarType m_threshold; // ν in [NJWS] Equation 5.52
  explicit RestartStrategy(ScalarType threshold = 0.1)
      : m_threshold(threshold) {}
  virtual ~RestartStrategy() = default;
  virtual bool restart(const ConjugacyContext &context) const = 0;
};

} // namespace nlcg
} // namespace optimize
} // namespace xts
