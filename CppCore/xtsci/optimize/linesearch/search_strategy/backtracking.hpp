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
namespace search_strategy {
template <typename ScalarType>
class BacktrackingSearch : public LineSearchStrategy<ScalarType> {
  std::reference_wrapper<LineSearchCondition<ScalarType>> m_cond;
  std::reference_wrapper<StepSizeStrategy<ScalarType>> m_step_strategy;
  ScalarType m_alpha_lo, m_alpha_hi;

public:
  explicit BacktrackingSearch(
      LineSearchCondition<ScalarType> &cond,
      StepSizeStrategy<ScalarType> &stepStrat, /* geomStep by default */
      ScalarType alpha_lo_val = 0.0, ScalarType alpha_hi_val = 1.0,
      OptimizeControl<ScalarType> optim = OptimizeControl<ScalarType>())
      : LineSearchStrategy<ScalarType>(optim), m_cond(cond),
        m_step_strategy(stepStrat), m_alpha_lo(alpha_lo_val),
        m_alpha_hi(alpha_hi_val) {}

  ScalarType search(const ObjectiveFunction<ScalarType> &func,
                    const SearchState<ScalarType> &cstate) override {
    ScalarType alpha_lo{m_alpha_lo}, alpha_hi{m_alpha_hi};
    ScalarType alpha = alpha_hi;
    while (!m_cond(alpha, func, cstate)) {
      alpha = m_step_strategy.get().nextStep(alpha_lo, alpha, func, cstate);
    }
    return alpha;
  }
};

} // namespace search_strategy
} // namespace linesearch
} // namespace optimize
} // namespace xts
