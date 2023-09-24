// MIT License
// Copyright 2023--present Rohit Goswami <HaoZeke>
#include "xtensor-io/xnpz.hpp"
#include "xtensor/xarray.hpp"

#include "include/xtensor_fmt.hpp"
#include "xtsci/optimize/linesearch/conditions/armijo.hpp"
#include "xtsci/optimize/linesearch/conditions/curvature.hpp"
#include "xtsci/optimize/linesearch/conditions/wolfe.hpp"
#include "xtsci/optimize/linesearch/search_strategy/backtracking.hpp"
#include "xtsci/optimize/trial_functions/quadratic.hpp"

#include <catch2/catch_all.hpp>

TEST_CASE("BacktrackingSearch Line Search", "[optimization]") {
  using ScalarType = double;
  xts::optimize::trial_functions::QuadraticFunction<ScalarType> quadratic;

  xt::xarray<ScalarType> x = xt::xarray<ScalarType>{-2.0}; // Starting point
  xt::xarray<ScalarType> direction =
      xt::xarray<ScalarType>{1.0}; // Move towards the minimum
  xts::optimize::SearchState<ScalarType> state{x, direction};

  xts::optimize::linesearch::conditions::ArmijoCondition<ScalarType>
      armijoCondition(0.0001);

  xts::optimize::linesearch::search_strategy::BacktrackingSearch<ScalarType>
      backtracking(armijoCondition);

  SECTION("BacktrackingSearch with Quadratic Function and Armijo Condition") {
    ScalarType alpha = backtracking.search(quadratic, state);
    REQUIRE(alpha > 0.0);
    REQUIRE(alpha <= 1.0);
  }

  SECTION("Using WeakWolfeCondition") {
    xts::optimize::linesearch::conditions::WeakWolfeCondition<double> condition;
    double alpha = backtracking.search(quadratic, {x, direction});
    REQUIRE(condition(alpha, quadratic, {x, direction}) == true);
  }

  SECTION("Using StrongWolfeCondition") {
    xts::optimize::linesearch::conditions::StrongWolfeCondition<double>
        condition;
    double alpha = backtracking.search(quadratic, {x, direction});
    REQUIRE(condition(alpha, quadratic, {x, direction}) == true);
  }

  SECTION("Using ArmijoCondition") {
    xts::optimize::linesearch::conditions::ArmijoCondition<double> condition;
    double alpha = backtracking.search(quadratic, {x, direction});
    REQUIRE(condition(alpha, quadratic, {x, direction}) == true);
  }

  SECTION("Using CurvatureCondition") {
    xts::optimize::linesearch::conditions::CurvatureCondition<double> condition;
    double alpha = backtracking.search(quadratic, {x, direction});
    REQUIRE(condition(alpha, quadratic, {x, direction}) == true);
  }

  SECTION("Using StrongCurvatureCondition") {
    xts::optimize::linesearch::conditions::StrongCurvatureCondition<double>
        condition;
    double alpha = backtracking.search(quadratic, {x, direction});
    REQUIRE(condition(alpha, quadratic, {x, direction}) == true);
  }
}

TEST_CASE("BacktrackingSearch Strategy Test with Different Beta",
          "[BacktrackingSearch]") {
  xts::optimize::trial_functions::QuadraticFunction<double> quadratic;

  xt::xarray<double> x = {1.0, 1.0};
  xt::xarray<double> direction = {-1.0, -1.0}; // Decreasing direction

  SECTION("Using WeakWolfeCondition with beta = 0.8") {
    xts::optimize::linesearch::conditions::WeakWolfeCondition<double> condition;
    xts::optimize::linesearch::search_strategy::BacktrackingSearch<double>
        backtracking(condition, 0.8);
    double alpha = backtracking.search(quadratic, {x, direction});
    REQUIRE(condition(alpha, quadratic, {x, direction}) == true);
  }
}

TEST_CASE("BacktrackingSearch Convergence Test", "[BacktrackingSearch]") {
  xts::optimize::trial_functions::QuadraticFunction<double> quadratic;

  xt::xarray<double> x = {1.0, 1.0};
  xt::xarray<double> direction = {-1.0, -1.0};
  xts::optimize::linesearch::conditions::WeakWolfeCondition<double> condition;
  xts::optimize::linesearch::search_strategy::BacktrackingSearch<double>
      backtracking(condition);

  SECTION("Using WeakWolfeCondition") {
    double alpha = backtracking.search(quadratic, {x, direction});

    REQUIRE(condition(alpha, quadratic, {x, direction}) == true);
    REQUIRE(quadratic(x + alpha * direction) < quadratic(x));
  }
}
