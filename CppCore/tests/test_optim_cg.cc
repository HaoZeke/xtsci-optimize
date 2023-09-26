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

#include "xtsci/optimize/minimize/nlcg.hpp"
#include "xtsci/optimize/nlcg/base.hpp"

#include "xtsci/optimize/trial_functions/quadratic.hpp"
#include "xtsci/optimize/trial_functions/rosenbrock.hpp"

#include "xtsci/optimize/nlcg/conjugacy/fletcher_reeves.hpp"
#include "xtsci/optimize/nlcg/conjugacy/polak_ribiere.hpp"

#include <catch2/catch_all.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

TEST_CASE("ConjugateGradientOptimizer with Backtracking Search") {
  xts::optimize::trial_functions::Rosenbrock<double> rosen;
  xts::optimize::OptimizeControl<double> control;
  xts::optimize::nlcg::conjugacy::FletcherReeves<double> fletcherreeves;
  xts::optimize::nlcg::conjugacy::PolakRibiere<double> polakribiere;
  control.tol = 1e-6;

  xt::xarray<double> initial_guess = {-1.3, 1.8};
  xt::xarray<double> direction = {0.0, 0.0};
  xts::optimize::SearchState<double> cstate = {initial_guess, direction};

  SECTION("With Armijo Condition") {
    xts::optimize::linesearch::conditions::ArmijoCondition<double> armijo(0.1);

    SECTION("Using Geometric Reduction Step Size") {
      xts::optimize::linesearch::step_size::GeometricReductionStepSize<double>
          geomStep;
      xts::optimize::linesearch::search_strategy::BacktrackingSearch<double>
          backtracking(armijo, geomStep);
      xts::optimize::minimize::ConjugateGradientOptimizer<double> optimizer(
          backtracking, fletcherreeves);

      xts::optimize::OptimizeResult<double> result =
          optimizer.optimize(rosen, cstate, control);

      REQUIRE_THAT(result.x(0), Catch::Matchers::WithinAbs(1.0, 1e-4));
      REQUIRE_THAT(result.x(1), Catch::Matchers::WithinAbs(1.0, 1e-4));

      REQUIRE(result.nit == 445);
    }

    SECTION("Using Bisection Step Size") {
      xts::optimize::linesearch::step_size::BisectionStepSize<double>
          bisectionStep;
      xts::optimize::linesearch::search_strategy::BacktrackingSearch<double>
          backtracking(armijo, bisectionStep);
      xts::optimize::minimize::ConjugateGradientOptimizer<double> optimizer(
          backtracking, fletcherreeves);

      xts::optimize::OptimizeResult<double> result =
          optimizer.optimize(rosen, cstate, control);

      REQUIRE_THAT(result.x(0), Catch::Matchers::WithinAbs(1.0, 1e-4));
      REQUIRE_THAT(result.x(1), Catch::Matchers::WithinAbs(1.0, 1e-4));

      REQUIRE(result.nit == 445);
    }

    SECTION("Using Golden Step Size") {
      xts::optimize::linesearch::step_size::GoldenStepSize<double> goldenStep;
      xts::optimize::linesearch::search_strategy::BacktrackingSearch<double>
          backtracking(armijo, goldenStep);
      xts::optimize::minimize::ConjugateGradientOptimizer<double> optimizer(
          backtracking, fletcherreeves);

      xts::optimize::OptimizeResult<double> result =
          optimizer.optimize(rosen, cstate, control);

      REQUIRE_THAT(result.x(0), Catch::Matchers::WithinAbs(1.0, 1e-4));
      REQUIRE_THAT(result.x(1), Catch::Matchers::WithinAbs(1.0, 1e-4));

      REQUIRE(result.nit == 85);
    }

    // SECTION("Using Cubic Step Size") {
    //   xts::optimize::linesearch::step_size::CubicStepSize<double> cubicStep;
    //   xts::optimize::linesearch::search_strategy::BacktrackingSearch<double>
    //       backtracking(armijo, cubicStep);
    //   xts::optimize::minimize::ConjugateGradientOptimizer<double> optimizer(
    //       backtracking);

    //   xts::optimize::OptimizeResult<double> result =
    //       optimizer.optimize(rosen, cstate, control);

    //   REQUIRE_THAT(result.x(0), Catch::Matchers::WithinAbs(1.0, 1e-4));
    //   REQUIRE_THAT(result.x(1), Catch::Matchers::WithinAbs(1.0, 1e-4));

    //   REQUIRE(result.nit == 300);
    // }
  }

  SECTION("With StrongWolfe Condition") {
    xts::optimize::linesearch::conditions::StrongWolfeCondition<double>
        strongwolfe(1e-4, 0.9);

    SECTION("Using Geometric Reduction Step Size") {
      xts::optimize::linesearch::step_size::GeometricReductionStepSize<double>
          geomStep;
      xts::optimize::linesearch::search_strategy::BacktrackingSearch<double>
          backtracking(strongwolfe, geomStep);
      xts::optimize::minimize::ConjugateGradientOptimizer<double> optimizer(
          backtracking, fletcherreeves);

      xts::optimize::OptimizeResult<double> result =
          optimizer.optimize(rosen, cstate, control);

      REQUIRE_THAT(result.x(0), Catch::Matchers::WithinAbs(1.0, 1e-4));
      REQUIRE_THAT(result.x(1), Catch::Matchers::WithinAbs(1.0, 1e-4));

      REQUIRE(result.nit == 90); // As per your previous test
    }

    SECTION("Using Bisection Step Size") {
      xts::optimize::linesearch::step_size::BisectionStepSize<double>
          bisectionStep;
      xts::optimize::linesearch::search_strategy::BacktrackingSearch<double>
          backtracking(strongwolfe, bisectionStep);
      xts::optimize::minimize::ConjugateGradientOptimizer<double> optimizer(
          backtracking, fletcherreeves);

      xts::optimize::OptimizeResult<double> result =
          optimizer.optimize(rosen, cstate, control);

      REQUIRE_THAT(result.x(0), Catch::Matchers::WithinAbs(1.0, 1e-4));
      REQUIRE_THAT(result.x(1), Catch::Matchers::WithinAbs(1.0, 1e-4));

      REQUIRE(result.nit == 90);
    }

    SECTION("Using Golden Step Size") {
      xts::optimize::linesearch::step_size::GoldenStepSize<double> goldenStep;
      xts::optimize::linesearch::search_strategy::BacktrackingSearch<double>
          backtracking(strongwolfe, goldenStep);
      xts::optimize::minimize::ConjugateGradientOptimizer<double> optimizer(
          backtracking, fletcherreeves);

      xts::optimize::OptimizeResult<double> result =
          optimizer.optimize(rosen, cstate, control);

      REQUIRE_THAT(result.x(0), Catch::Matchers::WithinAbs(1.0, 1e-4));
      REQUIRE_THAT(result.x(1), Catch::Matchers::WithinAbs(1.0, 1e-4));

      REQUIRE(result.nit == 85);
    }

    // SECTION("Using Cubic Step Size") {
    //   xts::optimize::linesearch::step_size::CubicStepSize<double> cubicStep;
    //   xts::optimize::linesearch::search_strategy::BacktrackingSearch<double>
    //       backtracking(strongwolfe, cubicStep);
    //   xts::optimize::minimize::ConjugateGradientOptimizer<double> optimizer(
    //       backtracking);

    //   xts::optimize::OptimizeResult<double> result =
    //       optimizer.optimize(rosen, cstate, control);

    //   REQUIRE_THAT(result.x(0), Catch::Matchers::WithinAbs(1.0, 1e-4));
    //   REQUIRE_THAT(result.x(1), Catch::Matchers::WithinAbs(1.0, 1e-4));

    //   REQUIRE(result.nit == 300);
    // }
  }

  SECTION("With Goldstein Condition") {
    xts::optimize::linesearch::conditions::GoldsteinCondition<double> goldstein(
        1e-4, 0.4);

    SECTION("Using Geometric Reduction Step Size") {
      xts::optimize::linesearch::step_size::GeometricReductionStepSize<double>
          geomStep;
      xts::optimize::linesearch::search_strategy::BacktrackingSearch<double>
          backtracking(goldstein, geomStep);
      xts::optimize::minimize::ConjugateGradientOptimizer<double> optimizer(
          backtracking, fletcherreeves);

      xts::optimize::OptimizeResult<double> result =
          optimizer.optimize(rosen, cstate, control);

      REQUIRE_THAT(result.x(0), Catch::Matchers::WithinAbs(1.0, 1e-4));
      REQUIRE_THAT(result.x(1), Catch::Matchers::WithinAbs(1.0, 1e-4));

      REQUIRE(result.nit == 455);
    }

    SECTION("Using Bisection Step Size") {
      xts::optimize::linesearch::step_size::BisectionStepSize<double>
          bisectionStep;
      xts::optimize::linesearch::search_strategy::BacktrackingSearch<double>
          backtracking(goldstein, bisectionStep);
      xts::optimize::minimize::ConjugateGradientOptimizer<double> optimizer(
          backtracking, fletcherreeves);

      xts::optimize::OptimizeResult<double> result =
          optimizer.optimize(rosen, cstate, control);

      REQUIRE_THAT(result.x(0), Catch::Matchers::WithinAbs(1.0, 1e-4));
      REQUIRE_THAT(result.x(1), Catch::Matchers::WithinAbs(1.0, 1e-4));

      REQUIRE(result.nit == 455);
    }

    SECTION("Using Golden Step Size") {
      xts::optimize::linesearch::step_size::GoldenStepSize<double> goldenStep;
      xts::optimize::linesearch::search_strategy::BacktrackingSearch<double>
          backtracking(goldstein, goldenStep);
      xts::optimize::minimize::ConjugateGradientOptimizer<double> optimizer(
          backtracking, fletcherreeves);

      xts::optimize::OptimizeResult<double> result =
          optimizer.optimize(rosen, cstate, control);

      REQUIRE_THAT(result.x(0), Catch::Matchers::WithinAbs(1.0, 1e-4));
      REQUIRE_THAT(result.x(1), Catch::Matchers::WithinAbs(1.0, 1e-4));

      REQUIRE(result.nit == 177);
    }

    // SECTION("Using Cubic Step Size") {
    //   xts::optimize::linesearch::step_size::CubicStepSize<double> cubicStep;
    //   xts::optimize::linesearch::search_strategy::BacktrackingSearch<double>
    //       backtracking(goldstein, cubicStep);
    //   xts::optimize::minimize::ConjugateGradientOptimizer<double> optimizer(
    //       backtracking);

    //   xts::optimize::OptimizeResult<double> result =
    //       optimizer.optimize(rosen, cstate, control);

    //   REQUIRE_THAT(result.x(0), Catch::Matchers::WithinAbs(1.0, 1e-4));
    //   REQUIRE_THAT(result.x(1), Catch::Matchers::WithinAbs(1.0, 1e-4));

    //   REQUIRE(result.nit == 300);
    // }
  }
}
