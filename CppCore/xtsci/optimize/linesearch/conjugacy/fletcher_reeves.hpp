#pragma once
// MIT License
// Copyright 2023--present Rohit Goswami <HaoZeke>
#include <algorithm>
#include <functional>
#include <limits>
#include <string>
#include <vector>

#include "xtensor-blas/xlinalg.hpp"

#include "xtsci/optimize/linesearch/base.hpp"

namespace xts {
namespace optimize {
namespace linesearch {
namespace conjugacy {
template <typename ScalarType>
class FletcherReeves : public ConjugacyCoefficientStrategy<ScalarType> {
public:
  ScalarType
  computeBeta(const xt::xarray<ScalarType> &current_gradient,
              const xt::xarray<ScalarType> &previous_gradient) const override {
    return xt::linalg::dot(current_gradient, current_gradient)() /
           xt::linalg::dot(previous_gradient, previous_gradient)();
  }
};
} // namespace conjugacy
} // namespace linesearch
} // namespace optimize
} // namespace xts
