#pragma once
// MIT License
// Copyright 2023--present Rohit Goswami <HaoZeke>
#include <algorithm>
#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "xtsci/optimize/base.hpp"
#include "xtsci/optimize/linesearch/base.hpp"
#include "xtsci/optimize/linesearch/step_size/geom.hpp"

namespace xts {
namespace optimize {
namespace linesearch {
namespace search_strategy {
template <typename ScalarType>
class BacktrackingSearch : public LineSearchStrategy<ScalarType> {
  std::reference_wrapper<LineSearchCondition<ScalarType>> m_cond;
  step_size::GeometricReductionStepSize<ScalarType> m_geom;

public:
  explicit BacktrackingSearch(
      LineSearchCondition<ScalarType> &cond, ScalarType geom_beta = 0.5,
      OptimizeControl<ScalarType> optim = OptimizeControl<ScalarType>())
      : LineSearchStrategy<ScalarType>(optim), m_cond(cond),
        m_geom{step_size::GeometricReductionStepSize<ScalarType>(geom_beta)} {}

  ScalarType search(const AlphaState<ScalarType> _in,
                    const ObjectiveFunction<ScalarType> &func,
                    const SearchState<ScalarType> &cstate) override {
    auto in_alpha = _in;
    ScalarType alpha = _in.init;
    while (alpha > 0 && !m_cond(alpha, func, cstate)) {
      alpha = m_geom.nextStep(in_alpha, func, cstate);
      in_alpha.init = alpha;
    }
    return alpha;
  }
};

} // namespace search_strategy
} // namespace linesearch
} // namespace optimize
} // namespace xts
