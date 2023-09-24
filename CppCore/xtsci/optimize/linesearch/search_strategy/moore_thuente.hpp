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
class MooreThuenteLineSearch : public LineSearchStrategy<ScalarType> {
private:
  conditions::ArmijoCondition<ScalarType> armijo;
  conditions::StrongCurvatureCondition<ScalarType> strong_curvature;
  std::reference_wrapper<StepSizeStrategy<ScalarType>> m_step_strategy;
  ScalarType m_alpha_lo, m_alpha_hi, m_alpha;
  ScalarType c_armijo, c_curv;

public:
  MooreThuenteLineSearch(
      StepSizeStrategy<ScalarType> &stepStrat, ScalarType c_armijo = 1e-4,
      ScalarType c_curv = 0.9,
      OptimizeControl<ScalarType> optim = OptimizeControl<ScalarType>(),
      ScalarType alpha_lo_val = 0.0, ScalarType alpha_hi_val = 1.0,
      ScalarType alpha_val = 0.5)
      : LineSearchStrategy<ScalarType>(optim), armijo(c_armijo),
        strong_curvature(c_curv), m_step_strategy(stepStrat),
        m_alpha_lo(alpha_lo_val), m_alpha_hi(alpha_hi_val), m_alpha(alpha_val),
        c_armijo(c_armijo), c_curv(c_curv) {}

  ScalarType search(const ObjectiveFunction<ScalarType> &func,
                    const SearchState<ScalarType> &cstate) override {
    ScalarType alpha_lo{m_alpha_lo}, alpha_hi{m_alpha_hi}, alpha{m_alpha};
    while (true) {
      auto grad = *func.gradient(cstate.x + alpha * cstate.direction);
      ScalarType grad_dir = xt::linalg::dot(grad, cstate.direction)();

      if (!(armijo(alpha, func, cstate))) {
        alpha_hi = alpha;
      } else if (grad_dir > 0) {
        alpha_hi = alpha;
      } else if (std::abs(grad_dir) <=
                 -c_curv * xt::linalg::dot(*func.gradient(cstate.x),
                                           cstate.direction)()) {
        alpha_lo = alpha;
      } else {
        // Both Strong Wolfe conditions are satisfied
        return alpha;
      }
      if (alpha_hi - alpha_lo < this->m_control.tol) {
        break;
      }
      // Update alpha using interpolation or other step strategy
      alpha = m_step_strategy.get().nextStep(alpha_lo, alpha_hi, func, cstate);
    }
    return alpha;
  }
};

} // namespace search_strategy
} // namespace linesearch
} // namespace optimize
} // namespace xts
