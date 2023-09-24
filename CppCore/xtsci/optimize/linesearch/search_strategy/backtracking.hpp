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
  ScalarType beta;
  std::reference_wrapper<LineSearchCondition<ScalarType>> m_cond;

public:
  explicit BacktrackingSearch(
      LineSearchCondition<ScalarType> &cond, ScalarType b = 0.5,
      OptimizeControl<ScalarType> optim = OptimizeControl<ScalarType>())
      : LineSearchStrategy<ScalarType>(optim), m_cond(cond), beta(b) {}

  ScalarType search(const ObjectiveFunction<ScalarType> &func,
                    const SearchState<ScalarType> &cstate) override {
    auto [x, direction] = cstate;
    ScalarType alpha = 1.0;
    while (!m_cond(alpha, func, cstate)) {
      alpha *= beta;
    }
    return alpha;
  }
};

} // namespace search_strategy
} // namespace linesearch
} // namespace optimize
} // namespace xts
