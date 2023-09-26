#pragma once
// MIT License
// Copyright 2023--present Rohit Goswami <HaoZeke>
#include <algorithm>
#include <functional>
#include <limits>
#include <string>
#include <vector>

#include "xtensor-blas/xlinalg.hpp"

#include "xtsci/optimize/linesearch/base.hpp"

namespace xts {
namespace optimize {
namespace linesearch {
namespace conjugacy {
template <typename ScalarType>
class LiuStorey : public ConjugacyCoefficientStrategy<ScalarType> {
public:
  ScalarType
  computeBeta(const ConjugacyContext<ScalarType> &ctx) const override {
    // [ZJJS] Equation 3, [LYCS] Equation 10
    auto grad_change = ctx.current_gradient - ctx.previous_gradient;
    return -1 *
           (xt::linalg::dot(ctx.current_gradient, grad_change)() /
            xt::linalg::dot(ctx.previous_direction, ctx.previous_gradient)());
  }

  // References:
  // [ZJJS] Shi, Zhen-Jun, and Jie Shen. “Convergence of Liu–Storey Conjugate
  // Gradient Method.” European Journal of Operational Research 182, no. 2
  // (October 16, 2007): 552–60. https://doi.org/10.1016/j.ejor.2006.09.066.
  //
  // [LYCS]  Liu, Y., and C. Storey. “Efficient Generalized Conjugate Gradient
  // Algorithms, Part 1: Theory.” Journal of Optimization Theory and
  // Applications 69, no. 1 (April 1, 1991): 129–37.
  // https://doi.org/10.1007/BF00940464.
};
} // namespace conjugacy
} // namespace linesearch
} // namespace optimize
} // namespace xts
