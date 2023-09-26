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

} // namespace nlcg
} // namespace optimize
} // namespace xts
