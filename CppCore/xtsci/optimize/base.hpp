#pragma once
// MIT License
// Copyright 2023--present Rohit Goswami <HaoZeke>
#include <algorithm>
#include <functional>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "xtensor-blas/xlinalg.hpp"
#include "xtensor/xarray.hpp"

#include "xtsci/func/base.hpp"

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
  size_t nufg;      // number of unique function and gradient evaluations
  size_t nit;       // number of iterations performed by the optimizer
  ScalarType maxcv; // the maximum constraint violation
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
  optimize(const func::ObjectiveFunction<ScalarType> &func,
           const SearchState<ScalarType> &initial_guess,
           const OptimizeControl<ScalarType> &control) const = 0;
};

} // namespace optimize
} // namespace xts
