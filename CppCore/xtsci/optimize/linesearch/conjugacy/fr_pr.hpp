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

#include "xtsci/optimize/linesearch/conjugacy/fletcher_reeves.hpp"
#include "xtsci/optimize/linesearch/conjugacy/polak_ribiere.hpp"

namespace xts {
namespace optimize {
namespace linesearch {
namespace conjugacy {
template <typename ScalarType>
class FRPR : public ConjugacyCoefficientStrategy<ScalarType> {
  FletcherReeves<ScalarType> m_fr;
  PolakRibiere<ScalarType> m_pr;

public:
  FRPR<ScalarType>()
      : m_fr{FletcherReeves<ScalarType>()}, m_pr{PolakRibiere<ScalarType>()} {}
  ScalarType
  computeBeta(const ConjugacyContext<ScalarType> &ctx) const override {
    // [NJWS] Equation 5.50
    auto beta_pr = m_pr.computeBeta(ctx);
    auto beta_fr = m_fr.computeBeta(ctx);
    if (beta_pr < -beta_fr) {
      return -beta_fr;
    } else if (std::abs(beta_pr) <= beta_fr) {
      return beta_pr;
    } else /* beta_pr > beta_fr */ {
      return beta_fr;
    }
  }

  // References:
  // [NJWS] Nocedal, J., & Wright, S. (2006). Numerical optimization. Springer
};
} // namespace conjugacy
} // namespace linesearch
} // namespace optimize
} // namespace xts
