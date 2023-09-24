#pragma once
// MIT License
// Copyright 2023--present Rohit Goswami <HaoZeke>
#include <algorithm>
#include <functional>
#include <limits>
#include <string>
#include <vector>

#include "xtsci/optimize/base.hpp"

namespace xts {
namespace optimize {
namespace linesearch {

template <typename ScalarType> class LineSearchCondition {
public:
  virtual bool operator()(ScalarType alpha,
                          const ObjectiveFunction<ScalarType> &func,
                          const SearchState<ScalarType> &cstate) const = 0;
};

template <typename ScalarType> class LineSearchStrategy {
public:
  virtual ScalarType
  search(const ObjectiveFunction<ScalarType> &func,
         const SearchState<ScalarType> &cstate,
         const LineSearchCondition<ScalarType> &condition) = 0;
};

template <typename ScalarType>
class LineSearchOptimizer : public AbstractOptimizer<ScalarType> {
  virtual OptimizeResult<ScalarType>
  optimize(const ObjectiveFunction<ScalarType> &func,
           const xt::xexpression<ScalarType> &initial_guess,
           const OptimizeControl<ScalarType> &control) const = 0;

protected:
  const LineSearchCondition<ScalarType>
      m_ls_cond; // Condition for step size acceptance
  const LineSearchStrategy<ScalarType>
      m_ls_strat; // Strategy for finding optimal step size
};

} // namespace linesearch
} // namespace optimize
} // namespace xts
