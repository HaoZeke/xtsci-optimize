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
#include "xtsci/optimize/linesearch/step_size/geom.hpp"

namespace xts {
namespace optimize {
namespace linesearch {
namespace search_strategy {
class BacktrackingSearch : public SearchStrategy {
  std::reference_wrapper<SearchCondition> m_cond;
  step_size::GeometricReductionStepSize m_geom;

public:
  explicit BacktrackingSearch(SearchCondition &cond, ScalarType geom_beta = 0.5,
                              OptimizeControl optim = OptimizeControl())
      : SearchStrategy(optim), m_cond(cond),
        m_geom{step_size::GeometricReductionStepSize(geom_beta)} {}

  ScalarType search(const AlphaState _in, const Optimizable &optobj,
                    const SearchState &cstate) override {
    auto in_alpha = _in;
    ScalarType alpha = _in.init;
    while (alpha > 0 && !m_cond(alpha, optobj, cstate)) {
      alpha = m_geom.nextStep(in_alpha, optobj, cstate);
      in_alpha.init = alpha;
    }
    return alpha;
  }
};

} // namespace search_strategy
} // namespace linesearch
} // namespace optimize
} // namespace xts
