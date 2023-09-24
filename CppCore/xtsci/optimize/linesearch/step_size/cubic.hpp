#pragma once
// MIT License
// Copyright 2023--present Rohit Goswami <HaoZeke>
#include <stdexcept> // For std::invalid_argument

#include "xtensor-blas/xlinalg.hpp"
#include "xtsci/optimize/base.hpp"
#include "xtsci/optimize/linesearch/base.hpp"

namespace xts {
namespace optimize {
namespace linesearch {
namespace step_size {

template <typename ScalarType>
class CubicInterpolationStep : public StepSizeStrategy<ScalarType> {
public:
  ScalarType nextStep(ScalarType alpha_lo, ScalarType alpha_hi,
                      const ObjectiveFunction<ScalarType> &func,
                      const SearchState<ScalarType> &state) const override {
    ScalarType f_lo = func(state.x + alpha_lo * state.direction);
    ScalarType f_hi = func(state.x + alpha_hi * state.direction);

    auto grad_lo_opt = func.gradient(state.x + alpha_lo * state.direction);
    auto grad_hi_opt = func.gradient(state.x + alpha_hi * state.direction);

    if (!grad_lo_opt || !grad_hi_opt) {
      // Cannot perform cubic interpolation without gradients
      throw std::invalid_argument("CubicInterpolationStep requires gradients "
                                  "to be defined in the ObjectiveFunction.");
    }

    ScalarType df_lo = xt::linalg::dot(*grad_lo_opt, state.direction)();
    ScalarType df_hi = xt::linalg::dot(*grad_hi_opt, state.direction)();

    // Coefficients for the cubic interpolant
    ScalarType d1 = df_lo + df_hi - 3 * (f_lo - f_hi) / (alpha_lo - alpha_hi);
    ScalarType d2 = sign(alpha_hi - alpha_lo) * sqrt(d1 * d1 - df_lo * df_hi);
    ScalarType alpha_c = alpha_hi - (alpha_hi - alpha_lo) * (df_hi + d2 - d1) /
                                        (df_hi - df_lo + 2 * d2);

    return alpha_c;
  }

private:
  ScalarType sign(ScalarType x) const { return (x < 0) ? -1 : 1; }
};

} // namespace step_size
} // namespace linesearch
} // namespace optimize
} // namespace xts
