// MIT License
// Copyright 2023--present Rohit Goswami <HaoZeke>
#include <fmt/format.h>
#include <fmt/ostream.h>

#include <cmath>
#include <cstdlib>

#include <iostream>
#include <random>

#include "xtensor-fmt/misc.hpp"
#include "xtensor-io/xnpz.hpp"
#include "xtensor/xarray.hpp"

#include "xtensor/xbuilder.hpp"
#include "xtsci/optimize/base.hpp"
#include "xtsci/optimize/linesearch/conditions/armijo.hpp"
#include "xtsci/optimize/linesearch/conditions/goldstein.hpp"
#include "xtsci/optimize/linesearch/conditions/wolfe.hpp"

#include "xtsci/optimize/linesearch/search_strategy/backtracking.hpp"
#include "xtsci/optimize/linesearch/search_strategy/zoom.hpp"

#include "xtsci/optimize/linesearch/step_size/hermite.hpp"
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

// #include "xtsci/optimize/minimize/adam.hpp"
// #include "xtsci/optimize/minimize/bfgs.hpp"
#include "xtsci/optimize/minimize/lbfgs.hpp"
// #include "xtsci/optimize/minimize/nlcg.hpp"
// #include "xtsci/optimize/minimize/pso.hpp"
// #include "xtsci/optimize/minimize/sd.hpp"
// #include "xtsci/optimize/minimize/sr1.hpp"
// #include "xtsci/optimize/minimize/sr2.hpp"

#include "xtsci/func/plot_aid.hpp"
#include "xtsci/func/trial/D2/branin.hpp"
#include "xtsci/func/trial/D2/eggholder.hpp"
#include "xtsci/func/trial/D2/himmelblau.hpp"
#include "xtsci/func/trial/D2/mullerbrown.hpp"
#include "xtsci/func/trial/D2/rosenbrock.hpp"

#include "rgpot/CuH2/CuH2Pot.hpp"
#include "xtsci/pot/base.hpp"

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
  xts::func::trial::D2::Rosenbrock<double> rosen;
  xts::func::trial::D2::Himmelblau<double> himmelblau;
  // xts::func::trial::D2::QuadraticFunction<double> quadratic;
  xts::func::trial::D2::Eggholder<double> eggholder;
  xts::func::trial::D2::MullerBrown<double> mullerbrown;
  xts::func::trial::D2::Branin<double> branin;

  xts::optimize::OptimizeControl control;
  control.tol = 1e-6;
  control.gtol = 1e-3;
  control.xtol = 1e-8;
  control.ftol = 1e-22;
  control.max_iterations = 10000;
  control.maxmove = 0.1;
  control.verbose = true;

  xts::optimize::linesearch::conditions::ArmijoCondition armijo(0.1);
  xts::optimize::linesearch::conditions::StrongWolfeCondition strongwolfe(1e-4,
                                                                          0.9);
  xts::optimize::linesearch::conditions::GoldsteinCondition goldstein(1e-2,
                                                                      1e-4);

  xts::optimize::linesearch::step_size::BisectionStepSize bisectionStep;
  xts::optimize::linesearch::step_size::GoldenStepSize goldenStep;
  xts::optimize::linesearch::step_size::CubicInterpolationStepSize cubicStep;
  xts::optimize::linesearch::step_size::SecantStepSize secantStep;
  xts::optimize::linesearch::step_size::GeometricReductionStepSize geomStep(
      0.5);
  xts::optimize::linesearch::step_size::QuadraticInterpolationStepSize quadStep;
  xts::optimize::linesearch::step_size::HermiteInterpolationStepSize
      hermiteCubicStep;

  xts::optimize::linesearch::search_strategy::BacktrackingSearch backtracking(
      strongwolfe, 0.5);
  xts::optimize::linesearch::search_strategy::ZoomLineSearch zoom(
      hermiteCubicStep, 1e-4, 0.9, control);

  xts::optimize::nlcg::conjugacy::FletcherReeves fletcherreeves;
  xts::optimize::nlcg::conjugacy::PolakRibiere polakribiere;
  xts::optimize::nlcg::conjugacy::HestenesStiefel hestenesstiefel;
  xts::optimize::nlcg::conjugacy::LiuStorey liustorey;
  xts::optimize::nlcg::conjugacy::HybridizedConj hybrid_min(
      hestenesstiefel, polakribiere,
      [](double a, double b) -> double { return std::min(a, b); });
  xts::optimize::nlcg::conjugacy::FRPR frpr;
  xts::optimize::nlcg::conjugacy::HagerZhang hagerzhang;

  xts::optimize::nlcg::restart::NJWSRestart njws_restart;
  xts::optimize::nlcg::restart::NeverRestart never_restart;

  // xts::optimize::minimize::ConjugateGradientOptimizer cgopt(
  //     zoom, liustorey, njws_restart);

  // xts::optimize::minimize::SteepestDescentOptimizer
  // sdopt(backtracking);

  // xts::optimize::minimize::BFGSOptimizer bfgsopt(zoom);
  xts::optimize::minimize::LBFGSOptimizer lbfgsopt(zoom, 6);
  // xts::optimize::minimize::ADAMOptimizer adaopt(backtracking);
  // xts::optimize::minimize::SR1Optimizer sr1opt(zoom);
  // xts::optimize::minimize::SR2Optimizer sr2opt(zoom);
  // xts::optimize::minimize::PSOptim psopt(100, 0.5, 1.5, 1.5,
  // control);

  auto cuh2pot = std::make_shared<rgpot::CuH2Pot>();
  auto CuH2Obj = xts::pot::mk_xtpot_con("cuh2.con", cuh2pot);

  xt::xarray<double> initial_guess = {
      8.68229999999999968, 9.94699999999999918, 4.75760000000000094,
      7.94209999999999994, 9.94699999999999918, 4.75760000000000094}; // cuh2
  // xt::xarray<double> initial_guess = {-1.2, 1.0}; // rosen
  // xt::xarray<double> initial_guess = {-1.3, 1.8}; // rosen
  // xt::xarray<double> initial_guess = {0.0, 0.0}; // himmelblau
  // xt::xarray<double> initial_guess = {0.23007699, 0.20781567}; // mullerbrown
  xt::xarray<double> direction = xt::zeros_like(initial_guess);
  xts::optimize::SearchState cstate = {initial_guess, direction};
  // xts::optimize::OptimizeResult result = lbfgsopt.optimize(CuH2Obj, cstate);
  size_t for_n = 30;
  auto n_pos = lbfgsopt.step_from(CuH2Obj, cstate, for_n);
  std::cout << "Optimized x: " << n_pos << "\n"
            << "After " << for_n << " steps\n"
            << "Function: " << CuH2Obj(n_pos) << std::endl
            << "Gradient norm: " << xt::linalg::norm(*CuH2Obj.gradient(n_pos))
            << std::endl;

  // xts::func::npz_on_grid2D<double>({-1.5, 1.2, 400}, {-0.2, 2.0, 400},
  //                                  mullerbrown, "mullerbrown.npz");
  // xts::func::npz_on_grid2D<double>({-5, 18, 400}, {-5, 20, 400}, branin,
  //                                  "branin.npz");
  // xts::optimize::OptimizeResult<double> result =
  //     psopt.optimize(mullerbrown, {-512, -512}, {512, 512});

  // std::cout << "Optimized x: " << result.x << "\n";
  // std::cout << "Function value: " << result.fun << "\n";
  // std::cout << "Number of iterations: " << result.nit << "\n";
  // std::cout << "Number of function evaluations: " << result.nfev << "\n";
  // std::cout << "Number of gradient evaluations: " << result.njev << "\n";
  // std::cout << "Number of Hessian evaluations: " << result.nhev << "\n";
  // std::cout << "Unique function and gradient calls: " << result.nufg << "\n";
  return EXIT_SUCCESS;
}
