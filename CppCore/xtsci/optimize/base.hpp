#pragma once
// MIT License
// Copyright 2023--present Rohit Goswami <HaoZeke>
#include <algorithm>
#include <functional>
#include <limits>
#include <optional>
#include <string>
#include <vector>

#include "xtensor-blas/xlinalg.hpp"
#include "xtensor/xarray.hpp"

namespace xts {
namespace optimize {

template <typename ScalarType = double> struct OptimizeResult {
  xt::xarray<ScalarType> x; // solution tensor
  bool success;             // whether or not the optimizer exited successfully
  int status;               // termination status of the optimizer
  std::string message;      // description of the termination
  ScalarType fun;           // value of objective function at the solution
  xt::xarray<ScalarType> jac;      // value of the Jacobian at the solution
  xt::xarray<ScalarType> hess;     // value of the Hessian at the solution
  xt::xarray<ScalarType> hess_inv; // inverse of the Hessian at the solution
  size_t nfev;      // number of evaluations of the objective functions
  size_t njev;      // number of evaluations of the Jacobian
  size_t nhev;      // number of evaluations of the Hessian
  size_t nit;       // number of iterations performed by the optimizer
  ScalarType maxcv; // the maximum constraint violation
};

struct EvaluationCounter {
  size_t function_evals = 0;
  size_t gradient_evals = 0;
  size_t hessian_evals = 0;
};

template <typename ScalarType = double> class ObjectiveFunction {
public:
  ObjectiveFunction() = default;

  virtual ~ObjectiveFunction() = default;

  ScalarType operator()(const xt::xarray<ScalarType> &x) const {
    ++m_counter.function_evals;
    return this->compute(x);
  }

  virtual std::optional<xt::xarray<ScalarType>>
  gradient(const xt::xarray<ScalarType> &x) const {
    ++m_counter.gradient_evals;
    return this->compute_gradient(x);
  }

  virtual std::optional<xt::xarray<ScalarType>>
  hessian(const xt::xarray<ScalarType> &x) const {
    ++m_counter.hessian_evals;
    return this->compute_hessian(x);
  }

  ScalarType
  directional_derivative(const xt::xarray<ScalarType> &x,
                         const xt::xarray<ScalarType> &direction) const {
    auto grad_opt = gradient(x);
    if (!grad_opt) {
      throw std::runtime_error(
          "Gradient required for computing directional derivative.");
    }
    return xt::linalg::dot(*grad_opt, direction)();
  }

  EvaluationCounter evaluation_counts() const { return m_counter; }

private:
  mutable EvaluationCounter m_counter;

  virtual ScalarType compute(const xt::xarray<ScalarType> &x) const = 0;

  virtual std::optional<xt::xarray<ScalarType>>
  compute_gradient(const xt::xarray<ScalarType> &) const {
    return std::nullopt;
  }

  virtual std::optional<xt::xarray<ScalarType>>
  compute_hessian(const xt::xarray<ScalarType> &) const {
    return std::nullopt;
  }
};

template <typename ScalarType = double> struct OptimizeControl {
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

template <typename ScalarType = double> struct SearchState {
  xt::xarray<ScalarType> x;         // Current point
  xt::xarray<ScalarType> direction; // Current search direction
  SearchState(const xt::xarray<ScalarType> &x,
              const xt::xarray<ScalarType> &direction)
      : x(x), direction(direction) {}
};

template <typename ScalarType = double> class AbstractOptimizer {
public:
  // Virtual destructor
  virtual ~AbstractOptimizer() = default;

  // Pure virtual function for optimization
  virtual OptimizeResult<ScalarType>
  optimize(const ObjectiveFunction<ScalarType> &func,
           const SearchState<ScalarType> &initial_guess,
           const OptimizeControl<ScalarType> &control) const = 0;
};

} // namespace optimize
} // namespace xts
