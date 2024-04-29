#pragma once
// MIT License
// Copyright 2023--present Rohit Goswami <HaoZeke>
#include <algorithm>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "xtensor/xarray.hpp"

#include "xtsci/func/base.hpp"
#include "xtsci/optimize/numerics.hpp"

namespace xts {
namespace optimize {

struct OptimizeResult {
  ScalarVec x;           // solution tensor
  bool success;          // whether or not the optimizer exited successfully
  int status;            // termination status of the optimizer
  std::string message;   // description of the termination
  ScalarType fun;        // value of objective function at the solution
  ScalarMatrix jac;      // value of the Jacobian at the solution
  ScalarMatrix hess;     // value of the Hessian at the solution
  ScalarMatrix hess_inv; // inverse of the Hessian at the solution
  size_t nfev;           // number of evaluations of the objective functions
  size_t njev;           // number of evaluations of the Jacobian
  size_t nhev;           // number of evaluations of the Hessian
  size_t nufg;           // number of unique function and gradient evaluations
  size_t nit;            // number of iterations performed by the optimizer
  ScalarType maxcv;      // the maximum constraint violation
};

struct OptimizeControl {
  size_t max_iterations = 1000; // Maximum number of iterations
  double maxmove = 1000;        // Maximum step size
  ScalarType tol = 1e-6;        // Tolerance for termination
  bool verbose = false;         // Whether or not to print progress
  ScalarType xtol = 1e-6;       // Change in x threshold
  ScalarType ftol = 1e-6;       // Change in f(x) threshold
  ScalarType gtol = 1e-6;       // Change in f'(x) threshold
  OptimizeControl(const size_t miter_val, const ScalarType tol_val,
                  const bool verb_val)
      : max_iterations{miter_val}, tol{tol_val}, verbose{verb_val} {}
  OptimizeControl() {}
  // TODO(rg): Should have a TerminateStrategy or something here
};

struct SearchState {
  ScalarVec x;         // Current point
  ScalarVec direction; // Current search direction
  SearchState(const ScalarVec &x, const ScalarVec &direction)
      : x(x), direction(direction) {}
};

struct AlphaState {
  ScalarType init;
  ScalarType low;
  ScalarType hi;
};

class StepSizeStrategy {
public:
  virtual ScalarType nextStep(const AlphaState alpha, const FObjFunc &func,
                              const SearchState &cstate) const = 0;
};

class SearchCondition {
public:
  virtual bool operator()(ScalarType alpha, const FObjFunc &func,
                          const SearchState &cstate) const = 0;
};

class SearchStrategy {
protected:
  friend class AbstractOptimizer;
  OptimizeControl m_control;

public:
  explicit SearchStrategy(const OptimizeControl &control)
      : m_control(control) {}
  virtual ScalarType search(const AlphaState _in, const FObjFunc &func,
                            const SearchState &cstate) = 0;
};

class AbstractOptimizer {
public:
  // Virtual destructor for proper cleanup of derived classes
  virtual ~AbstractOptimizer() = default;
  explicit AbstractOptimizer(SearchStrategy &strategy)
      : m_strat(strategy), m_control{strategy.m_control} {}
  explicit AbstractOptimizer(SearchStrategy &strategy, SearchState initial)
      : AbstractOptimizer(strategy) {
    set_initial(initial);
  }

  void set_initial(const SearchState &initial) {
    std::unique_lock<std::mutex> lock(m_mutex);
    m_cur = std::make_unique<SearchState>(initial);
    m_next = std::make_unique<SearchState>(initial);
    lock.unlock();
  }

  virtual OptimizeResult optimize(const FObjFunc &func,
                                  const SearchState &state) {
    set_initial(state);
    if (m_control.get().verbose) {
      // Print the headers in the desired format
      std::cout << "       Step     Time       Energy       fmax\n";
    }
    while (m_result.nit < m_control.get().max_iterations &&
           !this->converged(state)) {
      this->step(func);
      m_result.nit++;
    }
    return get_result(func);
  }

  OptimizeResult get_result(const FObjFunc &func) const {
    m_result.x = m_next->x;
    m_result.fun = func(m_next->x);
    m_result.jac = *func.gradient(m_next->x);
    m_result.nfev = func.evaluation_counts().function_evals;
    m_result.njev = func.evaluation_counts().gradient_evals;
    m_result.nhev = func.evaluation_counts().hessian_evals;
    m_result.nufg = func.evaluation_counts().unique_func_grad;
    return m_result;
  }

  virtual void step(const FObjFunc &func) = 0;
  // TODO(rg): this is pointless, just modify maxmove
  ScalarVec step_from(FObjFunc &func, SearchState &state, size_t for_n = 1) {
    set_initial(state);
    if (m_control.get().verbose) {
      // Print the headers in the desired format
      std::cout << "       Step     Time       Energy       fmax\n";
    }
    while (m_result.nit < for_n && !this->converged(state)) {
      this->step(func);
      m_result.nit++;
    }
    return m_next->x;
  }

protected:
  std::unique_ptr<SearchState> m_cur, m_next;
  std::mutex m_mutex;
  std::reference_wrapper<SearchStrategy> m_strat;
  const std::reference_wrapper<OptimizeControl> m_control;
  mutable OptimizeResult m_result;

  // Method to check convergence (can be overridden for custom behavior)
  bool converged(const SearchState &state) const;
};

} // namespace optimize
} // namespace xts
