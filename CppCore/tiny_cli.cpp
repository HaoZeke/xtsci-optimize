// MIT License
// Copyright 2023--present Rohit Goswami <HaoZeke>
#include <fmt/format.h>
#include <fmt/ostream.h>

#include <cmath>
#include <cstdlib>

#include <iostream>

#include "include/xtensor_fmt.hpp"
#include "xtensor-io/xnpz.hpp"
#include "xtensor/xarray.hpp"

#include "xtsci/optimize/base.hpp"
#include "xtsci/optimize/linesearch/conditions/armijo.hpp"
#include "xtsci/optimize/linesearch/conditions/wolfe.hpp"
#include "xtsci/optimize/linesearch/search_strategy/backtracking.hpp"
#include "xtsci/optimize/linesearch/search_strategy/bisection.hpp"
#include "xtsci/optimize/linesearch/search_strategy/zoom.hpp"
#include "xtsci/optimize/linesearch/step_size/bisect.hpp"
#include "xtsci/optimize/linesearch/step_size/cubic.hpp"
#include "xtsci/optimize/minimize/cg.hpp"
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
  xts::optimize::linesearch::conditions::ArmijoCondition<double> armijo(0.1);
  xts::optimize::linesearch::conditions::StrongWolfeCondition<double>
      strongwolfe(0.1, 0.1);
  xts::optimize::linesearch::search_strategy::BisectionSearch<double> bisection(
      armijo, -1.5, 1.5);
  xts::optimize::linesearch::search_strategy::BacktrackingSearch<double>
      backtracking(armijo);
  xts::optimize::linesearch::step_size::BisectionStep<double> bisectionStep;
  xts::optimize::linesearch::step_size::CubicInterpolationStep<double>
      cubicStep;
  xts::optimize::linesearch::search_strategy::ZoomLineSearch<double> zoom(
      bisectionStep);
  xts::optimize::minimize::ConjugateGradientOptimizer<double> optimizer(zoom);

  xt::xarray<double> initial_guess = {3.0, 4.0};
  xt::xarray<double> direction = {-1.0, -1.0};
  xts::optimize::SearchState<double> cstate = {initial_guess, direction};
  xts::optimize::OptimizeResult<double> result =
      optimizer.optimize(rosen, cstate, control);

  std::cout << "Optimized x: " << result.x << "\n";
  std::cout << "Function value: " << result.fun << "\n";
  std::cout << "Number of iterations: " << result.nit << "\n";
  return EXIT_SUCCESS;
}
