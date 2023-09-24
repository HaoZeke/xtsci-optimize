#pragma once
// MIT License
// Copyright 2023--present Rohit Goswami <HaoZeke>
#include <algorithm>
#include <functional>
#include <limits>
#include <optional>
#include <string>
#include <vector>

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

template <typename ScalarType = double> class ObjectiveFunction {
public:
  virtual ~ObjectiveFunction() = default;

  virtual ScalarType operator()(const xt::xarray<ScalarType> &x) const = 0;

  virtual std::optional<xt::xarray<ScalarType>>
  gradient(const xt::xarray<ScalarType> &x) const {
    return std::nullopt;
  }

  virtual std::optional<xt::xarray<ScalarType>>
  hessian(const xt::xarray<ScalarType> &x) const {
    return std::nullopt;
  }
};

template <typename ScalarType = double> struct OptimizeControl {
  const size_t max_iterations = 1000; // Maximum number of iterations
  const ScalarType tol = 1e-6;        // Tolerance for termination
  const bool verbose = false;          // Whether or not to print progress
  // TODO: Should have a TerminateStrategy or something here
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
           const xt::xexpression<ScalarType> &initial_guess,
           const OptimizeControl<ScalarType> &control) const = 0;
};

} // namespace optimize
} // namespace xts
