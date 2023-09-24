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
class BisectionSearch : public LineSearchStrategy<ScalarType> {
  ScalarType alpha_min, alpha_max;
  using LineSearchStrategy<ScalarType>::m_control;
public:
  BisectionSearch(
      ScalarType alpha_min_val = 0.0, ScalarType alpha_max_val = 1.0,
      OptimizeControl<ScalarType> optim = OptimizeControl<ScalarType>())
      : LineSearchStrategy<ScalarType>(optim), alpha_min(alpha_min_val),
        alpha_max(alpha_max_val) {}

  ScalarType search(const ObjectiveFunction<ScalarType> &func,
                    const SearchState<ScalarType> &cstate,
                    const LineSearchCondition<ScalarType> &condition) override {
    auto [x, direction] = cstate;
    ScalarType alpha = (alpha_min + alpha_max) / 2.0;
    size_t iter = 0;
    while (alpha_max - alpha_min > m_control.tol) {
      if (iter > m_control.max_iterations) {
        break;
      }
      if (condition(alpha, func, {x, direction})) {
        alpha_min = alpha;
      } else {
        alpha_max = alpha;
      }
      alpha = (alpha_min + alpha_max) / 2.0;
      if (alpha == alpha_min || alpha == alpha_max) {
        break;
      }
      iter++;
    }
    return alpha;
  }
};
} // namespace search_strategy
} // namespace linesearch
} // namespace optimize
} // namespace xts
