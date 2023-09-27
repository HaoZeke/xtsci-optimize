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
#include "xtsci/optimize/linesearch/search_strategy/zoom.hpp"

#include "xtsci/optimize/linesearch/step_size/quadratic.hpp"
#include "xtsci/optimize/nlcg/conjugacy/fletcher_reeves.hpp"
#include "xtsci/optimize/nlcg/conjugacy/fr_pr.hpp"
#include "xtsci/optimize/nlcg/conjugacy/hager_zhang.hpp"
#include "xtsci/optimize/nlcg/conjugacy/hestenes-stiefel.hpp"
#include "xtsci/optimize/nlcg/conjugacy/hybridized_conj.hpp"
#include "xtsci/optimize/nlcg/conjugacy/liu_storey.hpp"
#include "xtsci/optimize/nlcg/conjugacy/polak_ribiere.hpp"

#include "xtsci/optimize/nlcg/restart/never.hpp"
#include "xtsci/optimize/nlcg/restart/njws.hpp"

#include "xtsci/optimize/linesearch/step_size/bisect.hpp"
#include "xtsci/optimize/linesearch/step_size/cubic.hpp"
#include "xtsci/optimize/linesearch/step_size/geom.hpp"
#include "xtsci/optimize/linesearch/step_size/golden.hpp"
#include "xtsci/optimize/linesearch/step_size/secant.hpp"

#include "xtsci/optimize/minimize/adam.hpp"
#include "xtsci/optimize/minimize/bfgs.hpp"
#include "xtsci/optimize/minimize/lbfgs.hpp"
#include "xtsci/optimize/minimize/nlcg.hpp"
#include "xtsci/optimize/minimize/pso.hpp"
#include "xtsci/optimize/minimize/sd.hpp"
#include "xtsci/optimize/minimize/sr1.hpp"
#include "xtsci/optimize/minimize/sr2.hpp"

#include "xtsci/optimize/trial_functions/eggholder.hpp"
#include "xtsci/optimize/trial_functions/himmelblau.hpp"
#include "xtsci/optimize/trial_functions/mullerbrown.hpp"
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
  xts::optimize::trial_functions::Himmelblau<double> himmelblau;
  xts::optimize::trial_functions::Eggholder<double> eggholder;
  xts::optimize::trial_functions::MullerBrown<double> mullerbrown;

  xts::optimize::OptimizeControl<double> control;
  control.tol = 1e-6;
  control.gtol = 1e-5;
  control.xtol = 1e-8;
  control.ftol = 1e-22;
  control.max_iterations = 10000;
  control.maxmove = 100;
  control.verbose = true;

  xts::optimize::linesearch::conditions::ArmijoCondition<double> armijo(0.1);
  xts::optimize::linesearch::conditions::StrongWolfeCondition<double>
      strongwolfe(1e-4, 0.9);
  xts::optimize::linesearch::conditions::GoldsteinCondition<double> goldstein(
      1e-2, 1e-4);

  xts::optimize::linesearch::step_size::BisectionStepSize<double> bisectionStep;
  xts::optimize::linesearch::step_size::GoldenStepSize<double> goldenStep;
  xts::optimize::linesearch::step_size::CubicInterpolationStepSize<double>
      cubicStep;
  xts::optimize::linesearch::step_size::SecantStepSize<double> secantStep;
  xts::optimize::linesearch::step_size::GeometricReductionStepSize<double>
      geomStep(0.5);
  xts::optimize::linesearch::step_size::QuadraticInterpolationStepSize<double>
      quadStep;

  xts::optimize::linesearch::search_strategy::BacktrackingSearch<double>
      backtracking(strongwolfe, 0.5);
  xts::optimize::linesearch::search_strategy::ZoomLineSearch<double> zoom(
      cubicStep, 1e-4, 0.9);

  xts::optimize::nlcg::conjugacy::FletcherReeves<double> fletcherreeves;
  xts::optimize::nlcg::conjugacy::PolakRibiere<double> polakribiere;
  xts::optimize::nlcg::conjugacy::HestenesStiefel<double> hestenesstiefel;
  xts::optimize::nlcg::conjugacy::LiuStorey<double> liustorey;
  xts::optimize::nlcg::conjugacy::HybridizedConj<double> hybrid_min(
      hestenesstiefel, polakribiere,
      [](double a, double b) -> double { return std::min(a, b); });
  xts::optimize::nlcg::conjugacy::FRPR<double> frpr;
  xts::optimize::nlcg::conjugacy::HagerZhang<double> hagerzhang;

  xts::optimize::nlcg::restart::NJWSRestart<double> njws_restart;
  xts::optimize::nlcg::restart::NeverRestart<double> never_restart;

  xts::optimize::minimize::ConjugateGradientOptimizer<double> cgopt(
      zoom, frpr, never_restart);

  xts::optimize::minimize::SteepestDescentOptimizer<double> sdopt(backtracking);

  xts::optimize::minimize::BFGSOptimizer<double> bfgsopt(zoom);
  xts::optimize::minimize::LBFGSOptimizer<double> lbfgsopt(zoom, 10);
  xts::optimize::minimize::ADAMOptimizer<double> adaopt(backtracking);
  xts::optimize::minimize::SR1Optimizer<double> sr1opt(zoom);
  xts::optimize::minimize::SR2Optimizer<double> sr2opt(zoom);
  xts::optimize::minimize::PSOptim<double> psopt(100, 0.5, 1.5, 1.5, control);

  xt::xarray<double> initial_guess = {-1.2, 1.0}; // rosen
  // xt::xarray<double> initial_guess = {-1.3, 1.8}; // rosen
  // xt::xarray<double> initial_guess = {0.0, 0.0}; // himmelblau
  // xt::xarray<double> initial_guess = {0.23007699, 0.20781567}; // mullerbrown
  xt::xarray<double> direction = {0.0, 0.0};
  xts::optimize::SearchState<double> cstate = {initial_guess, direction};
  xts::optimize::OptimizeResult<double> result =
      lbfgsopt.optimize(rosen, cstate, control);

  // xts::optimize::OptimizeResult<double> result =
  //     psopt.optimize(mullerbrown, {-512, -512}, {512, 512});

  std::cout << "Optimized x: " << result.x << "\n";
  std::cout << "Function value: " << result.fun << "\n";
  std::cout << "Number of iterations: " << result.nit << "\n";
  std::cout << "Number of function evaluations: " << result.nfev << "\n";
  std::cout << "Number of gradient evaluations: " << result.njev << "\n";
  std::cout << "Number of Hessian evaluations: " << result.nhev << "\n";
  return EXIT_SUCCESS;
}
