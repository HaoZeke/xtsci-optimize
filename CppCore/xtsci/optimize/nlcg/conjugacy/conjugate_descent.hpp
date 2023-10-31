#pragma once
// MIT License
// Copyright 2023--present Rohit Goswami <HaoZeke>
#include <algorithm>
#include <functional>
#include <limits>
#include <string>
#include <vector>

#include "xtensor-blas/xlinalg.hpp"

#include "xtsci/optimize/nlcg/base.hpp"

namespace xts {
namespace optimize {
namespace nlcg {
namespace conjugacy {
template <typename ScalarType>
class ConjugateDescent : public ConjugacyCoefficientStrategy<ScalarType> {
public:
  ScalarType
  computeBeta(const ConjugacyContext<ScalarType> &ctx) const override {
    // [ZJJS] Equation 3
    return (xt::linalg::dot(ctx.current_gradient, ctx.current_gradient)() /
            xt::linalg::dot(ctx.previous_direction, ctx.previous_gradient)());
  }

  // References:
  // [ZJJS] Shi, Zhen-Jun, and Jie Shen. “Convergence of Liu–Storey Conjugate
  // Gradient Method.” European Journal of Operational Research 182, no. 2
  // (October 16, 2007): 552–60. https://doi.org/10.1016/j.ejor.2006.09.066.
  //
  // [NJWS] Nocedal, J., & Wright, S. (2006). Numerical optimization. Springer
};

} // namespace conjugacy
} // namespace nlcg
} // namespace optimize
} // namespace xts
