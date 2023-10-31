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
template <typename ScalarType> struct ConjugacyContext {
  xt::xarray<ScalarType> current_gradient;
  xt::xarray<ScalarType> previous_gradient;
  xt::xarray<ScalarType> previous_direction;
};

template <typename ScalarType> class ConjugacyCoefficientStrategy {
public:
  virtual ~ConjugacyCoefficientStrategy() = default;

  virtual ScalarType
  computeBeta(const ConjugacyContext<ScalarType> &context) const = 0;
};

template <typename ScalarType> class RestartStrategy {
public:
  ScalarType m_threshold; // ν in [NJWS] Equation 5.52
  explicit RestartStrategy(ScalarType threshold = 0.1)
      : m_threshold(threshold) {}
  virtual ~RestartStrategy() = default;
  virtual bool restart(const ConjugacyContext<ScalarType> &context) const = 0;
};

} // namespace nlcg
} // namespace optimize
} // namespace xts
