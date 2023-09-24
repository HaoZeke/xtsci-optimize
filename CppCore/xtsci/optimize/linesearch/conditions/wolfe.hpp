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
#include "xtsci/optimize/linesearch/base.hpp"
#include "xtsci/optimize/linesearch/conditions/armijo.hpp"
#include "xtsci/optimize/linesearch/conditions/curvature.hpp"

namespace xts {
namespace optimize {
namespace linesearch {
namespace conditions {

template <typename ScalarType>
class WeakWolfeCondition : public LineSearchCondition<ScalarType> {
    ArmijoCondition<ScalarType> armijo;
    CurvatureCondition<ScalarType> curvature;

public:
    WeakWolfeCondition(ScalarType c_armijo = 0.0001, ScalarType c_curvature = 0.9)
        : armijo(c_armijo), curvature(c_curvature) {}

    bool operator()(ScalarType alpha,
                    const ObjectiveFunction<ScalarType>& func,
                    const SearchState<ScalarType>& cstate) const override {
        return armijo(alpha, func, cstate) && curvature(alpha, func, cstate);
    }
};

template <typename ScalarType>
class StrongWolfeCondition : public LineSearchCondition<ScalarType> {
    ArmijoCondition<ScalarType> armijo;
    StrongCurvatureCondition<ScalarType> curvature;

public:
    StrongWolfeCondition(ScalarType c_armijo = 0.0001, ScalarType c_curvature = 0.9)
        : armijo(c_armijo), curvature(c_curvature) {}

    bool operator()(ScalarType alpha,
                    const ObjectiveFunction<ScalarType>& func,
                    const SearchState<ScalarType>& cstate) const override {
        return armijo(alpha, func, cstate) && curvature(alpha, func, cstate);
    }
};

} // namespace conditions
} // namespace linesearch
} // namespace optimize
} // namespace xts
