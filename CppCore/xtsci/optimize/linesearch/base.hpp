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

template <typename ScalarType> struct AlphaState {
  ScalarType init;
  ScalarType low;
  ScalarType hi;
};

template <typename ScalarType> class StepSizeStrategy {
public:
  virtual ScalarType nextStep(const AlphaState<ScalarType> alpha,
                              const ObjectiveFunction<ScalarType> &func,
                              const SearchState<ScalarType> &cstate) const = 0;
};

template <typename ScalarType> class LineSearchCondition {
public:
  virtual bool operator()(ScalarType alpha,
                          const ObjectiveFunction<ScalarType> &func,
                          const SearchState<ScalarType> &cstate) const = 0;
};

template <typename ScalarType> class LineSearchStrategy {
protected:
  OptimizeControl<ScalarType> m_control;

public:
  explicit LineSearchStrategy(const OptimizeControl<ScalarType> &control)
      : m_control(control) {}
  virtual ScalarType search(const AlphaState<ScalarType> _in,
                            const ObjectiveFunction<ScalarType> &func,
                            const SearchState<ScalarType> &cstate) = 0;
};

template <typename ScalarType>
class LineSearchOptimizer : public AbstractOptimizer<ScalarType> {
  virtual OptimizeResult<ScalarType>
  optimize(const ObjectiveFunction<ScalarType> &func,
           const SearchState<ScalarType> &initial_guess,
           const OptimizeControl<ScalarType> &control) const = 0;

protected:
  LineSearchStrategy<ScalarType>
      &m_ls_strat; // Strategy for finding optimal step size

public:
  explicit LineSearchOptimizer(LineSearchStrategy<ScalarType> &strategy)
      : m_ls_strat(strategy) {}
};

} // namespace linesearch
} // namespace optimize
} // namespace xts
