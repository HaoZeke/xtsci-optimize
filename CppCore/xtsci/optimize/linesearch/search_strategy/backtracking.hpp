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

public:
  explicit BacktrackingSearch(
      ScalarType b = 0.5,
      OptimizeControl<ScalarType> optim = OptimizeControl<ScalarType>())
      : LineSearchStrategy<ScalarType>(optim), beta(b) {}

  ScalarType search(const ObjectiveFunction<ScalarType> &func,
                    const SearchState<ScalarType> &cstate,
                    const LineSearchCondition<ScalarType> &condition) override {
    auto [x, direction] = cstate;
    ScalarType alpha = 1.0;
    while (!condition(alpha, func, {x, direction})) {
      alpha *= beta;
    }
    return alpha;
  }
};

} // namespace search_strategy
} // namespace linesearch
} // namespace optimize
} // namespace xts
