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
namespace restart {
template <typename ScalarType>
class NeverRestart : public RestartStrategy<ScalarType> {
public:
  bool restart(const ConjugacyContext<ScalarType> &) const override {
    // Sometimes (testing purposes mostly) it makes no sense to ever restart
    return false;
  }
};

} // namespace restart
} // namespace nlcg
} // namespace optimize
} // namespace xts
