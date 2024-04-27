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
class HagerZhang : public ConjugacyCoefficientStrategy {
public:
  ScalarType computeBeta(const ConjugacyContext &ctx) const override {
    auto grad_change = ctx.current_gradient - ctx.previous_gradient;
    auto grad_change_norm_sq = xt::linalg::dot(grad_change, grad_change);
    auto grad_change_prev_dot =
        xt::linalg::dot(grad_change, ctx.previous_direction);
    // [NJWS] Equation 5.50, [WHHZ] Equation 1.3
    auto term1 = grad_change - (2 * ctx.previous_direction) *
                                   (grad_change_norm_sq / grad_change_prev_dot);
    auto term2 =
        (xt::linalg::dot(ctx.current_gradient, ctx.current_gradient)() /
         grad_change_prev_dot);
    return (term1 * term2)();
  }

  // References:
  // [WHHZ] W. W. HAGER AND H. ZHANG, A new conjugate gradient method with
  // guaranteed descent and an efficient line search, SIAM Journal on
  // Optimization, 16 (2005), pp. 170–192.
  //
  // [NJWS] Nocedal, J., & Wright, S. (2006). Numerical optimization. Springer
};

} // namespace conjugacy
} // namespace nlcg
} // namespace optimize
} // namespace xts
