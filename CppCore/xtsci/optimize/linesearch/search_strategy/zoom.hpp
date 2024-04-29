#pragma once
// MIT License
// Copyright 2023--present Rohit Goswami <HaoZeke>
#include <algorithm>
#include <functional>
#include <limits>
#include <string>
#include <vector>

#include "xtsci/optimize/base.hpp"
#include "xtsci/optimize/linesearch/conditions/armijo.hpp"
#include "xtsci/optimize/linesearch/conditions/curvature.hpp"
#include "xtsci/optimize/numerics.hpp"

namespace xts {
namespace optimize {
namespace linesearch {
namespace search_strategy {
class ZoomLineSearch : public SearchStrategy {
private:
  conditions::ArmijoCondition armijo;
  conditions::StrongCurvatureCondition strong_curvature;
  std::reference_wrapper<StepSizeStrategy> m_step_strategy;

public:
  ZoomLineSearch(StepSizeStrategy &stepStrat, ScalarType c_armijo = 1e-4,
                 ScalarType c_curv = 0.9,
                 OptimizeControl optim = OptimizeControl())
      : SearchStrategy(optim), armijo(c_armijo), strong_curvature(c_curv),
        m_step_strategy(stepStrat) {}

  ScalarType search(const AlphaState _in, const Optimizable &optobj,
                    const SearchState &cstate) {
    auto phi = [&](ScalarType a_val) {
      return optobj(cstate.x + a_val * cstate.direction);
    };
    auto phi_prime = [&](ScalarType a_val) {
      return optobj.directional_derivative(cstate.x + a_val * cstate.direction,
                                           cstate.direction);
    };

    ScalarType phi_0 = phi(0.0);
    ScalarType phi_prime_0 = phi_prime(0.0);

    ScalarType alpha_max = _in.hi;
    ScalarType alpha_i = _in.init;
    ScalarType alpha_prev = 0.0; // Initialization corrected
    ScalarType alpha_res = std::numeric_limits<ScalarType>::infinity();

    for (size_t idx = 0; idx < 100; idx++) {
      if ((!armijo(alpha_i, optobj, cstate) && idx > 0) ||
          phi(alpha_i) > phi_0 + armijo.c * alpha_i * phi_prime_0) {
        alpha_res = zoom(alpha_prev, alpha_i, optobj, cstate);
        break;
      }
      if (strong_curvature(alpha_i, optobj, cstate)) {
        alpha_res = alpha_i;
        break;
      }
      if (phi_prime(alpha_i) >= 0) {
        alpha_res = zoom(alpha_i, alpha_prev, optobj, cstate);
        break;
      }
      alpha_prev = alpha_i;
      alpha_i = std::min(alpha_i * 2, alpha_max);
    }

    if (alpha_res == std::numeric_limits<ScalarType>::infinity() ||
        std::isnan(alpha_res)) {
      fmt::print("Failure, falling back to bisection of original interval\n");
      alpha_res = (_in.hi + _in.low) / 2;
    }
    return alpha_res;
  }

  ScalarType zoom(ScalarType lo, ScalarType hi, const Optimizable &optobj,
                  const SearchState &cstate) {
    auto phi = [&](ScalarType a_val) {
      return optobj(cstate.x + a_val * cstate.direction);
    };

    auto phi_prime = [&](ScalarType a_val) {
      return optobj.directional_derivative(cstate.x + a_val * cstate.direction,
                                           cstate.direction);
    };

    ScalarType alpha_j;      // Uses m_step_strategy below
    ScalarType previous_phi; // Initialized at the end of the loop
    const ScalarType ftol = this->m_control.ftol;
    const ScalarType xtol = this->m_control.xtol;
    const size_t max_iterations = this->m_control.max_iterations;

    for (size_t idx = 0; idx < max_iterations; ++idx) {
      alpha_j = m_step_strategy.get().nextStep(
          {.init = alpha_j, .low = lo, .hi = hi}, optobj, cstate);

      ScalarType current_phi = phi(alpha_j);
      // If the interval is too small, or the optobjtion is flat, we are done
      if ((std::abs(current_phi - previous_phi) < ftol ||
           std::abs(hi - lo) < xtol) &&
          idx > 0) {
        break;
      }

      if (!armijo(alpha_j, optobj, cstate) || current_phi >= phi(lo)) {
        hi = alpha_j;
      } else {
        if (strong_curvature(alpha_j, optobj, cstate)) {
          return alpha_j;
        }
        if (phi_prime(alpha_j) * (hi - lo) >= 0) {
          hi = lo;
        }
        lo = alpha_j;
      }
      previous_phi = current_phi;
    }
    return m_step_strategy.get().nextStep(
        {.init = alpha_j, .low = lo, .hi = hi}, optobj, cstate);
  }
};

} // namespace search_strategy
} // namespace linesearch
} // namespace optimize
} // namespace xts
