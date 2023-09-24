#pragma once
// MIT License
// Copyright 2023--present Rohit Goswami <HaoZeke>
#include <algorithm>
#include <functional>
#include <limits>
#include <string>
#include <vector>

#include "xtsci/optimize/linesearch/base.hpp"

namespace xts {
namespace optimize {
namespace linesearch {
namespace search_strategy {
template <typename ScalarType>
class BisectionSearch : public LineSearchStrategy<ScalarType> {
    ScalarType alpha_min, alpha_max;

public:
    BisectionSearch(ScalarType alpha_min_val = 0.0, ScalarType alpha_max_val = 1.0)
        : alpha_min(alpha_min_val), alpha_max(alpha_max_val) {}

    ScalarType search(const ObjectiveFunction<ScalarType>& func,
                      const SearchState<ScalarType>& cstate,
                      const LineSearchCondition<ScalarType>& condition) override {
        auto [x, direction] = cstate;
        ScalarType alpha = (alpha_min + alpha_max) / 2.0;
        while (alpha_max - alpha_min > 1e-5) {
            if (condition(alpha, func, { x, direction })) {
                alpha_min = alpha;
            } else {
                alpha_max = alpha;
            }
            alpha = (alpha_min + alpha_max) / 2.0;
        }
        return alpha;
    }
};
}
} // namespace linesearch
} // namespace optimize
} // namespace xts
