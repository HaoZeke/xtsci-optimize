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
#include "xtsci/optimize/linesearch/conditions/armijo.hpp"
#include "xtsci/optimize/linesearch/conditions/curvature.hpp"

namespace xts {
namespace optimize {
namespace linesearch {
namespace search_strategy {
template <typename ScalarType>
class ZoomLineSearch : public LineSearchStrategy<ScalarType> {
private:
  conditions::ArmijoCondition<ScalarType> armijo;
  conditions::StrongCurvatureCondition<ScalarType> strong_curvature;

public:
  ZoomLineSearch(
      ScalarType c_armijo = 0.01, ScalarType c_curv = 0.9,
      OptimizeControl<ScalarType> optim = OptimizeControl<ScalarType>())
      : LineSearchStrategy<ScalarType>(optim), armijo(c_armijo),
        strong_curvature(c_curv) {}

  ScalarType search(const ObjectiveFunction<ScalarType> &func,
                    const SearchState<ScalarType> &cstate) override {
    ScalarType alpha_lo = 0.0;
    ScalarType alpha_hi = 1.0;
    ScalarType alpha =
        0.5 * (alpha_lo + alpha_hi); // Start with a midpoint guess
    auto [x, direction] = cstate;
    while (true) {
      if (!(armijo(alpha, func, {x, direction}))) {
        alpha_hi = alpha;
      } else if (!(strong_curvature(alpha, func, {x, direction}))) {
        alpha_lo = alpha;
      } else {
        // Both conditions are satisfied
        return alpha;
      }
      if (alpha_hi - alpha_lo < this->m_control.tol) {
        break;
      }
      alpha = 0.5 * (alpha_lo + alpha_hi); // Update alpha to be the midpoint
      // TODO(rg): Use more sophisticated methods for Zoom
    }
    return alpha;
  }
};

} // namespace search_strategy
} // namespace linesearch
} // namespace optimize
} // namespace xts
