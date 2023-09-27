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
#include "xtsci/optimize/linesearch/step_size/bisect.hpp"

namespace xts {
namespace optimize {
namespace linesearch {
namespace search_strategy {
template <typename ScalarType>
class ZoomLineSearch : public LineSearchStrategy<ScalarType> {
private:
  conditions::ArmijoCondition<ScalarType> armijo;
  conditions::StrongCurvatureCondition<ScalarType> strong_curvature;
  std::reference_wrapper<StepSizeStrategy<ScalarType>> m_step_strategy;

public:
  ZoomLineSearch(
      StepSizeStrategy<ScalarType> &stepStrat, ScalarType c_armijo = 1e-4,
      ScalarType c_curv = 0.9,
      OptimizeControl<ScalarType> optim = OptimizeControl<ScalarType>())
      : LineSearchStrategy<ScalarType>(optim), armijo(c_armijo),
        strong_curvature(c_curv), m_step_strategy(stepStrat) {}

  ScalarType search(const AlphaState<ScalarType> _in,
                    const ObjectiveFunction<ScalarType> &func,
                    const SearchState<ScalarType> &cstate) {
    auto phi = [&](ScalarType a_val) {
      return func(cstate.x + a_val * cstate.direction);
    };
    auto phi_prime = [&](ScalarType a_val) {
      return func.directional_derivative(cstate.x + a_val * cstate.direction,
                                         cstate.direction);
    };

    ScalarType phi_0 = phi(0.0);
    ScalarType phi_prime_0 = phi_prime(0.0);

    ScalarType alpha_max = _in.hi;
    ScalarType alpha_i = _in.init;
    ScalarType alpha_prev = 0.0; // Initialization corrected
    ScalarType alpha_res = std::numeric_limits<ScalarType>::infinity();

    for (size_t idx = 0; idx < 100; idx++) {
      if ((!armijo(alpha_i, func, cstate) && idx > 0) ||
          phi(alpha_i) > phi_0 + armijo.c * alpha_i * phi_prime_0) {
        alpha_res = zoom(alpha_prev, alpha_i, func, cstate);
        break;
      }
      if (strong_curvature(alpha_i, func, cstate)) {
        alpha_res = alpha_i;
        break;
      }
      if (phi_prime(alpha_i) >= 0) {
        alpha_res = zoom(alpha_i, alpha_prev, func, cstate);
        break;
      }
      alpha_prev = alpha_i;
      alpha_i = std::min(alpha_i * 2, alpha_max);
    }

    if (alpha_res == std::numeric_limits<ScalarType>::infinity() ||
        std::isnan(alpha_res)) {
      fmt::print("Failure, falling back\n");
      alpha_res = std::min(alpha_i, _in.low);
    }
    return alpha_res;
  }

  ScalarType zoom(ScalarType lo, ScalarType hi,
                  const ObjectiveFunction<ScalarType> &func,
                  const SearchState<ScalarType> &cstate) {
    auto phi = [&](ScalarType a_val) {
      return func(cstate.x + a_val * cstate.direction);
    };

    auto phi_prime = [&](ScalarType a_val) {
      return func.directional_derivative(cstate.x + a_val * cstate.direction,
                                         cstate.direction);
    };

    ScalarType alpha_j;

    for (size_t idx = 0; idx < 100; ++idx) {
      alpha_j = m_step_strategy.get().nextStep(
          {.init = alpha_j, .low = lo, .hi = hi}, func, cstate);

      if (!armijo(alpha_j, func, cstate) || phi(alpha_j) >= phi(lo)) {
        hi = alpha_j;
      } else {
        if (strong_curvature(alpha_j, func, cstate)) {
          return alpha_j;
        }
        if (phi_prime(alpha_j) * (hi - lo) >= 0) {
          hi = lo;
        }
        lo = alpha_j;
      }
    }
    return m_step_strategy.get().nextStep(
        {.init = alpha_j, .low = lo, .hi = hi}, func, cstate);
  }
};

} // namespace search_strategy
} // namespace linesearch
} // namespace optimize
} // namespace xts
