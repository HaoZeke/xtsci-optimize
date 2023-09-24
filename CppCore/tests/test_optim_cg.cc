// MIT License
// Copyright 2023--present Rohit Goswami <HaoZeke>
#include "xtensor-io/xnpz.hpp"
#include "xtensor/xarray.hpp"

#include "include/xtensor_fmt.hpp"

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

#include "xtsci/optimize/minimize/cg.hpp"

#include "xtsci/optimize/trial_functions/quadratic.hpp"
#include "xtsci/optimize/trial_functions/rosenbrock.hpp"

#include <catch2/catch_all.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

TEST_CASE("ConjugateGradientOptimizer with Backtracking Search") {
  xts::optimize::trial_functions::Rosenbrock<double> rosen;

  xts::optimize::OptimizeControl<double> control;
  control.tol = 1e-6;

  xts::optimize::linesearch::conditions::StrongWolfeCondition<double>
      strongwolfe(1e-4, 0.9);
  xts::optimize::linesearch::step_size::GeometricReductionStepSize<double>
      geomStep;
  xts::optimize::linesearch::search_strategy::BacktrackingSearch<double>
      backtracking(strongwolfe, geomStep);

  xts::optimize::minimize::ConjugateGradientOptimizer<double> optimizer(
      backtracking);

  xt::xarray<double> initial_guess = {-1.3, 1.8};
  xt::xarray<double> direction = {0.0, 0.0};
  xts::optimize::SearchState<double> cstate = {initial_guess, direction};
  xts::optimize::OptimizeResult<double> result =
      optimizer.optimize(rosen, cstate, control);

  REQUIRE_THAT(result.x(0), Catch::Matchers::WithinAbs(1.0, 1e-4));
  REQUIRE_THAT(result.x(1), Catch::Matchers::WithinAbs(1.0, 1e-4));

  REQUIRE(result.nit == 90);
}
