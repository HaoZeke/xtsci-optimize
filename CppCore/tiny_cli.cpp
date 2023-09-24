// MIT License
// Copyright 2023--present Rohit Goswami <HaoZeke>
#include <fmt/format.h>
#include <fmt/ostream.h>

#include <cmath>
#include <cstdlib>

#include <iostream>
#include <random>

#include "include/xtensor_fmt.hpp"
#include "xtensor-io/xnpz.hpp"
#include "xtensor/xarray.hpp"

#include "xtsci/optimize/base.hpp"
#include "xtsci/optimize/linesearch/conditions/armijo.hpp"
#include "xtsci/optimize/linesearch/conditions/goldstein.hpp"
#include "xtsci/optimize/linesearch/conditions/wolfe.hpp"

#include "xtsci/optimize/linesearch/search_strategy/backtracking.hpp"
#include "xtsci/optimize/linesearch/search_strategy/moore_thuente.hpp"
#include "xtsci/optimize/linesearch/search_strategy/zoom.hpp"

#include "xtsci/optimize/linesearch/step_size/bisect.hpp"
#include "xtsci/optimize/linesearch/step_size/cubic.hpp"
#include "xtsci/optimize/linesearch/step_size/geom.hpp"
#include "xtsci/optimize/linesearch/step_size/golden.hpp"

#include "xtsci/optimize/minimize/adam.hpp"
#include "xtsci/optimize/minimize/bfgs.hpp"
#include "xtsci/optimize/minimize/cg.hpp"
#include "xtsci/optimize/minimize/lbfgs.hpp"
#include "xtsci/optimize/minimize/sr1.hpp"
#include "xtsci/optimize/minimize/sr2.hpp"

#include "xtsci/optimize/trial_functions/quadratic.hpp"
#include "xtsci/optimize/trial_functions/rosenbrock.hpp"

int main(int argc, char *argv[]) {
  // Eat warnings, also safer
  static_cast<void>(argc);
  static_cast<void>(argv);
  // // Define the grid
  // size_t n_points = 100; // Number of points along each axis
  // double x_min = -2.0, x_max = 2.0;
  // double y_min = -2.0, y_max = 2.0;

  // // Generate grid
  // xt::xtensor<double, 1> x = xt::linspace<double>(-2.0, 2.0, 100);
  // xt::xtensor<double, 1> y = xt::linspace<double>(-2.0, 2.0, 100);

  // xt::xtensor<double, 2> X, Y;
  // std::tie(X, Y) = xt::meshgrid(x, y);

  // // Evaluate Rosenbrock function on the grid
  // xt::xtensor<double, 2> Z = rosenbrock<double>(X, Y);

  // // Save to disk
  // // Write data to NPZ
  // xt::dump_npz("rosenbrock.npz", "X", X);
  // xt::dump_npz("rosenbrock.npz", "Y", Y,
  //              true); // The 'true' means append to existing file
  // xt::dump_npz("rosenbrock.npz", "Z", Z, true);

  // std::cout << "Data written to rosenbrock.npz" << std::endl;

  // Use a minimizer
  xts::optimize::trial_functions::Rosenbrock<double> rosen;
  xts::optimize::trial_functions::QuadraticFunction<double> quadratic;

  xts::optimize::OptimizeControl<double> control;
  control.tol = 1e-6;
  control.verbose = true;

  xts::optimize::linesearch::conditions::ArmijoCondition<double> armijo(0.1);
  xts::optimize::linesearch::conditions::StrongWolfeCondition<double>
      strongwolfe(1e-4, 0.9);
  xts::optimize::linesearch::conditions::GoldsteinCondition<double> goldstein(
      1e-2, 1e-4);

  xts::optimize::linesearch::step_size::BisectionStepSize<double> bisectionStep;
  xts::optimize::linesearch::step_size::GoldenStepSize<double> goldenStep;
  xts::optimize::linesearch::step_size::CubicStepSize<double> cubicStep;
  xts::optimize::linesearch::step_size::GeometricReductionStepSize<double>
      geomStep;

  xts::optimize::linesearch::search_strategy::BacktrackingSearch<double>
      backtracking(strongwolfe, goldenStep);
  xts::optimize::linesearch::search_strategy::ZoomLineSearch<double> zoom(
      bisectionStep, 1e-4, 0.9);
  xts::optimize::linesearch::search_strategy::MooreThuenteLineSearch<double>
      moorethuente(goldenStep, 1e-3, 0.3);

  xts::optimize::minimize::ConjugateGradientOptimizer<double> cgopt(
      backtracking);
  xts::optimize::minimize::BFGSOptimizer<double> bfgsopt(backtracking);
  xts::optimize::minimize::LBFGSOptimizer<double> lbfgsopt(zoom, 30);
  xts::optimize::minimize::ADAMOptimizer<double> adaopt(backtracking);
  xts::optimize::minimize::SR1Optimizer<double> sr1opt(zoom);
  xts::optimize::minimize::SR2Optimizer<double> sr2opt(zoom);

  xt::xarray<double> initial_guess = {-1.3, 1.8};
  xt::xarray<double> direction = {0.0, 0.0};
  xts::optimize::SearchState<double> cstate = {initial_guess, direction};
  xts::optimize::OptimizeResult<double> result =
      lbfgsopt.optimize(rosen, cstate, control);

  std::cout << "Optimized x: " << result.x << "\n";
  std::cout << "Function value: " << result.fun << "\n";
  std::cout << "Number of iterations: " << result.nit << "\n";
  std::cout << "Number of function evaluations: " << result.nfev << "\n";
  std::cout << "Number of gradient evaluations: " << result.njev << "\n";
  std::cout << "Number of Hessian evaluations: " << result.nhev << "\n";
  return EXIT_SUCCESS;
}
