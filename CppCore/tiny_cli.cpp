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

#include "xtsci/optimize/minimize/adam.hpp"
#include "xtsci/optimize/minimize/bfgs.hpp"
#include "xtsci/optimize/minimize/lbfgs.hpp"
#include "xtsci/optimize/minimize/nlcg.hpp"
#include "xtsci/optimize/minimize/pso.hpp"
#include "xtsci/optimize/minimize/sd.hpp"
#include "xtsci/optimize/minimize/sr1.hpp"
#include "xtsci/optimize/minimize/sr2.hpp"

#include "xtsci/func/plot_aid.hpp"
#include "xtsci/func/trial/D2/branin.hpp"
#include "xtsci/func/trial/D2/eggholder.hpp"
#include "xtsci/func/trial/D2/himmelblau.hpp"
#include "xtsci/func/trial/D2/mullerbrown.hpp"
#include "xtsci/func/trial/D2/rosenbrock.hpp"

#include "rgpot/CuH2/CuH2Pot.hpp"
#include "xtsci/pot/base.hpp"

#include "Helpers.hpp"
#include "include/BaseTypes.hpp"
#include "include/FormatConstants.hpp"
#include "include/ReadCon.hpp"
#include "include/helpers/StringHelpers.hpp"

xt::xtensor<double, 2>
extract_positions(const yodecon::types::ConFrameVec &frame) {
  size_t n_atoms = frame.x.size();
  std::array<size_t, 2> shape = {static_cast<size_t>(n_atoms), 3};

  xt::xtensor<double, 2> positions = xt::empty<double>(shape);
  for (size_t i = 0; i < n_atoms; ++i) {
    positions(i, 0) = frame.x[i];
    positions(i, 1) = frame.y[i];
    positions(i, 2) = frame.z[i];
  }

  return positions;
}

xt::xtensor<double, 1> normalize(const xt::xtensor<double, 1> &vec) {
  double norm = xt::linalg::norm(vec);
  if (norm == 0.0) {
    throw std::runtime_error("Cannot normalize a zero vector");
  }
  return vec / norm;
}

xt::xtensor<double, 2>
peturb_positions(const xt::xtensor<double, 2> &base_positions,
                 const xt::xtensor<int, 1> &atmNumVec, double hcu_dist,
                 double hh_dist) {
  xt::xtensor<double, 2> positions = base_positions;
  std::vector<size_t> hIndices, cuIndices;

  for (size_t i = 0; i < atmNumVec.size(); ++i) {
    if (atmNumVec(i) == 1) { // Hydrogen atom
      hIndices.push_back(i);
    } else if (atmNumVec(i) == 29) { // Copper atom
      cuIndices.push_back(i);
    } else {
      throw std::runtime_error("Unexpected atomic number");
    }
  }

  if (hIndices.size() != 2) {
    throw std::runtime_error("Expected exactly two hydrogen atoms");
  }

  // Compute the midpoint of the hydrogens
  auto hMidpoint =
      (xt::row(positions, hIndices[0]) + xt::row(positions, hIndices[1])) / 2;

  // TODO(rg): This is buggy in cuh2vizR!! (maybe)
  // Compute the HH direction
  xt::xtensor<double, 1> hh_direction;
  size_t h1_idx, h2_idx;
  if (positions(hIndices[0], 0) < positions(hIndices[1], 0)) {
    hh_direction = normalize(xt::row(positions, hIndices[1]) -
                             xt::row(positions, hIndices[0]));
    h1_idx = hIndices[0];
    h2_idx = hIndices[1];
  } else {
    hh_direction = normalize(xt::row(positions, hIndices[0]) -
                             xt::row(positions, hIndices[1]));
    h1_idx = hIndices[1];
    h2_idx = hIndices[0];
  }

  // Set the new position of the hydrogens using the recorded indices
  xt::row(positions, h1_idx) = hMidpoint - (0.5 * hh_dist) * hh_direction;
  xt::row(positions, h2_idx) = hMidpoint + (0.5 * hh_dist) * hh_direction;

  // Find the z-coordinate of the topmost Cu layer
  double maxCuZ = std::numeric_limits<double>::lowest();
  for (auto cuIndex : cuIndices) {
    maxCuZ = std::max(maxCuZ, positions(cuIndex, 2));
  }

  // Compute the new z-coordinate for the H atoms
  double new_z = maxCuZ + hcu_dist;

  // Update the z-coordinates of the H atoms
  for (auto hIndex : hIndices) {
    positions(hIndex, 2) = new_z;
  }

  return positions;
}

std::pair<double, double>
calculateDistances(const xt::xtensor<double, 2> &positions,
                   const xt::xtensor<int, 1> &atmNumVec) {
  std::vector<size_t> hIndices, cuIndices;
  for (size_t i = 0; i < atmNumVec.size(); ++i) {
    if (atmNumVec(i) == 1) { // Hydrogen atom
      hIndices.push_back(i);
    } else if (atmNumVec(i) == 29) { // Copper atom
      cuIndices.push_back(i);
    } else {
      throw std::runtime_error("Unexpected atomic number");
    }
  }

  if (hIndices.size() != 2) {
    throw std::runtime_error("Expected exactly two hydrogen atoms");
  }

  // Calculate the distance between Hydrogen atoms
  double hDistance =
      xt::linalg::norm(xt::view(positions, hIndices[0], xt::all()) -
                       xt::view(positions, hIndices[1], xt::all()));

  // Calculate the midpoint of Hydrogen atoms
  xt::xtensor<double, 1> hMidpoint =
      (xt::view(positions, hIndices[0], xt::all()) +
       xt::view(positions, hIndices[1], xt::all())) /
      2.0;

  // Find the z-coordinate of the topmost Cu layer
  double maxCuZ = std::numeric_limits<double>::lowest();
  for (size_t cuIndex : cuIndices) {
    maxCuZ = std::max(maxCuZ, positions(cuIndex, 2));
  }

  double cuSlabDist = positions(hIndices[0], 2) - maxCuZ;

  return std::make_pair(hDistance, cuSlabDist);
}

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

  xts::optimize::OptimizeControl<double> control;
  control.tol = 1e-5;
  control.gtol = 1e-5;
  control.xtol = 1e-8;
  control.ftol = 1e-22;
  control.max_iterations = 10000;
  control.maxmove = 10000;
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
  xts::optimize::linesearch::step_size::HermiteInterpolationStepSize<double>
      hermiteCubicStep;

  xts::optimize::linesearch::search_strategy::BacktrackingSearch<double>
      backtracking(strongwolfe, 0.5);
  xts::optimize::linesearch::search_strategy::ZoomLineSearch<double> zoom(
      cubicStep, 1e-4, 0.9, control);

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
      zoom, liustorey, njws_restart);

  // xts::optimize::minimize::SteepestDescentOptimizer<double>
  // sdopt(backtracking);

  xts::optimize::minimize::BFGSOptimizer<double> bfgsopt(zoom);
  xts::optimize::minimize::LBFGSOptimizer<double> lbfgsopt(zoom, 100);
  // xts::optimize::minimize::ADAMOptimizer<double> adaopt(backtracking);
  // xts::optimize::minimize::SR1Optimizer<double> sr1opt(zoom);
  // xts::optimize::minimize::SR2Optimizer<double> sr2opt(zoom);
  // xts::optimize::minimize::PSOptim<double> psopt(100, 0.5, 1.5, 1.5,
  // control);

  auto cuh2pot = std::make_shared<rgpot::CuH2Pot>();
  std::vector<std::string> fconts =
      yodecon::helpers::file::read_con_file("pos.con");
  auto frame = yodecon::create_single_con<yodecon::types::ConFrameVec>(fconts);

  auto positions = extract_positions(frame);
  auto atomNumbersVec = yodecon::symbols_to_atomic_numbers(frame.symbol);
  xt::xtensor<int, 1> atomTypes = xt::empty<int>({atomNumbersVec.size()});
  for (size_t i = 0; i < atomNumbersVec.size(); ++i) {
    atomTypes(i) = atomNumbersVec[i];
  }
  // std::array<size_t, 2> shape = {1, 3};
  xt::xtensor<double, 2> boxMatrix = xt::empty<double>(xt::shape({1, 3}));
  for (size_t i = 0; i < 3; ++i) {
    boxMatrix(0, i) = frame.boxl[i];
  }

  xts::pot::XTPot<double> objcuh2(cuh2pot, atomTypes, boxMatrix, xt::adapt(frame.is_fixed));
  auto energyFunc = [&objcuh2, &positions, &atomTypes](
                        double hh_dist, double cu_slab_dist) -> double {
    auto perturbed_positions =
        peturb_positions(positions, atomTypes, cu_slab_dist, hh_dist);
    return objcuh2(xt::ravel<xt::layout_type::row_major>(perturbed_positions)) -
           (-697.311695);
  };

  // xts::func::npz_on_grid2D<double>({0.4, 3.2, 60}, {-0.05, 3.1, 60},
  // energyFunc,
  //                                  "cuh2_grid.npz");

  // xt::xarray<double> initial_guess = {-1.2, 1.0}; // rosen
  // xt::xarray<double> initial_guess = {-1.3, 1.8}; // rosen
  // xt::xarray<double> initial_guess = {0.0, 0.0}; // himmelblau
  // xt::xarray<double> initial_guess = {0.23007699, 0.20781567}; // mullerbrown
  // xt::xarray<double> direction = {0.0, 0.0};
  // Flatten the positions to match what objcuh2.compute() expects
  xt::xtensor<double, 1> initial_guess =
      xt::ravel<xt::layout_type::row_major>(positions);
  xt::xtensor<double, 1> direction = xt::zeros_like(initial_guess);
  xts::optimize::SearchState<double> cstate = {initial_guess, direction};
  xts::optimize::OptimizeResult<double> result =
      lbfgsopt.optimize(objcuh2, cstate, control);

  // xts::func::npz_on_grid2D<double>({-1.5, 1.2, 400}, {-0.2, 2.0, 400},
  //                                  mullerbrown, "mullerbrown.npz");
  // xts::func::npz_on_grid2D<double>({-5, 18, 400}, {-5, 20, 400}, branin,
  //                                  "branin.npz");
  // xts::optimize::OptimizeResult<double> result =
  //     psopt.optimize(mullerbrown, {-512, -512}, {512, 512});

  // std::cout << "Optimized x: " << result.x << "\n";
  // std::cout << "Gradients: " << *objcuh2.gradient(result.x) << "\n";
  std::cout << "Function value: " << result.fun << "\n";
  std::cout << "Number of iterations: " << result.nit << "\n";
  std::cout << "Number of function evaluations: " << result.nfev << "\n";
  std::cout << "Number of gradient evaluations: " << result.njev << "\n";
  std::cout << "Number of Hessian evaluations: " << result.nhev << "\n";
  std::cout << "Unique function and gradient calls: " << result.nufg << "\n";
  return EXIT_SUCCESS;
}
