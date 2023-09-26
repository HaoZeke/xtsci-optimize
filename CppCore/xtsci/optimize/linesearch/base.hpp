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
protected:
  OptimizeControl<ScalarType> m_control;

public:
  explicit LineSearchStrategy(const OptimizeControl<ScalarType> &control)
      : m_control(control) {}
  virtual ScalarType search(const ObjectiveFunction<ScalarType> &func,
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

template <typename ScalarType> class StepSizeStrategy {
public:
  virtual ScalarType nextStep(ScalarType alpha_lo, ScalarType alpha_hi,
                              const ObjectiveFunction<ScalarType> &func,
                              const SearchState<ScalarType> &state) const = 0;
};

template <typename ScalarType> struct ConjugacyContext {
  xt::xarray<ScalarType> current_gradient;
  xt::xarray<ScalarType> previous_gradient;
  xt::xarray<ScalarType> current_direction;
  xt::xarray<ScalarType> previous_direction;
};

template <typename ScalarType> class ConjugacyCoefficientStrategy {
public:
  virtual ~ConjugacyCoefficientStrategy() = default;

  virtual ScalarType
  computeBeta(const ConjugacyContext<ScalarType> &context) const = 0;
};

} // namespace linesearch
} // namespace optimize
} // namespace xts
