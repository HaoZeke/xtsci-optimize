#pragma once
// MIT License
// Copyright 2023--present Rohit Goswami <HaoZeke>
#include <algorithm>
#include <functional>
#include <limits>
#include <optional>
#include <string>
#include <vector>

#include "xtensor-blas/xlinalg.hpp"

#include "xtsci/optimize/base.hpp"
#include "xtsci/optimize/linesearch/conditions/armijo.hpp"
#include "xtsci/optimize/linesearch/conditions/curvature.hpp"
#include "xtsci/optimize/numerics.hpp"

namespace xts {
namespace optimize {
namespace linesearch {
namespace conditions {

class WeakWolfeCondition : public SearchCondition {
  ArmijoCondition armijo;
  CurvatureCondition curvature;

public:
  explicit WeakWolfeCondition(ScalarType c_armijo = 1e-4,
                              ScalarType c_curvature = 0.9)
      : armijo(c_armijo), curvature(c_curvature) {}

  bool operator()(ScalarType alpha, const FObjFunc &func,
                  const SearchState &cstate) const override {
    return armijo(alpha, func, cstate) && curvature(alpha, func, cstate);
  }
};

class StrongWolfeCondition : public SearchCondition {
  ArmijoCondition armijo;
  StrongCurvatureCondition curvature;

public:
  explicit StrongWolfeCondition(ScalarType c_armijo = 1e-4,
                                ScalarType c_curvature = 0.9)
      : armijo(c_armijo), curvature(c_curvature) {}

  bool operator()(ScalarType alpha, const FObjFunc &func,
                  const SearchState &cstate) const override {
    return armijo(alpha, func, cstate) && curvature(alpha, func, cstate);
  }
};

} // namespace conditions
} // namespace linesearch
} // namespace optimize
} // namespace xts
