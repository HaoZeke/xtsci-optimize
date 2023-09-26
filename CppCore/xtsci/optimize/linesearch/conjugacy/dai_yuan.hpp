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
class ConjugateDescent : public ConjugacyCoefficientStrategy<ScalarType> {
public:
  ScalarType
  computeBeta(const ConjugacyContext<ScalarType> &ctx) const override {
    auto grad_change = ctx.current_gradient - ctx.previous_gradient;
    // [ZJJS] Equation 3
    return (xt::linalg::dot(ctx.current_gradient, ctx.current_gradient)() /
            xt::linalg::dot(grad_change, ctx.previous_direction)());
  }
};

// References:
// [ZJJS] Shi, Zhen-Jun, and Jie Shen. Convergence of Liu–Storey Conjugate
// Gradient Method. European Journal of Operational Research 182, no. 2
// (October 16, 2007): 552–60. https://doi.org/10.1016/j.ejor.2006.09.066.
//
// [DY] Y.H. Dai, Y. Yuan, A nonlinear conjugate gradient method with a strong
// global convergence property, SIAM Journal of Optimization 10 (1999) 177–182

} // namespace conjugacy
} // namespace linesearch
} // namespace optimize
} // namespace xts
