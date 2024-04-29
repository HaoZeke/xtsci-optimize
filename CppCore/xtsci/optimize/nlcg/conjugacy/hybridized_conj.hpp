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

class HybridizedConj : public ConjugacyCoefficientStrategy {
  std::reference_wrapper<ConjugacyCoefficientStrategy> m_ccs1, m_ccs2;
  std::function<ScalarType(ScalarType, ScalarType)> m_operator;

public:
  HybridizedConj(
      ConjugacyCoefficientStrategy &ccs1, ConjugacyCoefficientStrategy &ccs2,
      std::function<ScalarType(ScalarType, ScalarType)> op =
          [](ScalarType a, ScalarType b) -> ScalarType {
        return std::max(a, b);
      })
      : m_ccs1{ccs1}, m_ccs2{ccs2}, m_operator{op} {}

  ScalarType computeBeta(const ConjugacyContext &ctx) const override {
    return m_operator(m_ccs1.get().computeBeta(ctx),
                      m_ccs2.get().computeBeta(ctx));
  }
};

} // namespace conjugacy
} // namespace nlcg
} // namespace optimize
} // namespace xts
