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
class NeverRestart : public RestartStrategy {
public:
  bool restart(const ConjugacyContext &) const override {
    // Sometimes (testing purposes mostly) it makes no sense to ever restart
    // Also for the FRPR which basically has its own restart strategy
    return false;
  }
};

} // namespace restart
} // namespace nlcg
} // namespace optimize
} // namespace xts
