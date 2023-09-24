// MIT License
// Copyright 2023--present Rohit Goswami <HaoZeke>
#include "xtensor-io/xnpz.hpp"
#include "xtensor/xarray.hpp"

#include "include/xtensor_fmt.hpp"
#include "xtsci/optimize/linesearch/conditions/armijo.hpp"
#include "xtsci/optimize/linesearch/conditions/curvature.hpp"
#include "xtsci/optimize/linesearch/conditions/wolfe.hpp"
#include "xtsci/optimize/linesearch/search_strategy/bisection.hpp"
#include "xtsci/optimize/trial_functions/quadratic.hpp"

#include <catch2/catch_all.hpp>

TEST_CASE("BisectionSearch Strategy Test", "[BisectionSearch]") {
  xts::optimize::trial_functions::QuadraticFunction<double> quadratic;

  xt::xarray<double> x = {1.0, 1.0};
  xt::xarray<double> direction = {-1.0, -1.0}; // Decreasing direction

  SECTION("Using WeakWolfeCondition") {
    xts::optimize::linesearch::conditions::WeakWolfeCondition<double> condition;
    xts::optimize::linesearch::search_strategy::BisectionSearch<double>
        bisection(condition);
    double alpha = bisection.search(quadratic, {x, direction});
    // Let's check if the found alpha satisfies the condition
    REQUIRE(condition(alpha, quadratic, {x, direction}) == true);
  }

  SECTION("Using StrongWolfeCondition") {
    xts::optimize::linesearch::conditions::StrongWolfeCondition<double>
        condition;
    xts::optimize::linesearch::search_strategy::BisectionSearch<double>
        bisection(condition);
    double alpha = bisection.search(quadratic, {x, direction});
    // Let's check if the found alpha satisfies the condition
    REQUIRE(condition(alpha, quadratic, {x, direction}) == true);
  }

  SECTION("Using ArmijoCondition") {
    xts::optimize::linesearch::conditions::ArmijoCondition<double> condition;
    xts::optimize::linesearch::search_strategy::BisectionSearch<double>
        bisection(condition);
    double alpha = bisection.search(quadratic, {x, direction});
    // Let's check if the found alpha satisfies the condition
    REQUIRE(condition(alpha, quadratic, {x, direction}) == true);
  }

  SECTION("Using CurvatureCondition") {
    xts::optimize::linesearch::conditions::CurvatureCondition<double> condition;
    xts::optimize::linesearch::search_strategy::BisectionSearch<double>
        bisection(condition);
    double alpha = bisection.search(quadratic, {x, direction});
    // Let's check if the found alpha satisfies the condition
    REQUIRE(condition(alpha, quadratic, {x, direction}) == true);
  }

  SECTION("Using StrongCurvatureCondition") {
    xts::optimize::linesearch::conditions::StrongCurvatureCondition<double>
        condition;
    xts::optimize::linesearch::search_strategy::BisectionSearch<double>
        bisection(condition);
    double alpha = bisection.search(quadratic, {x, direction});
    // Let's check if the found alpha satisfies the condition
    REQUIRE(condition(alpha, quadratic, {x, direction}) == true);
  }
}
TEST_CASE("BisectionSearch Strategy Test with Different Initial Bounds",
          "[BisectionSearch]") {
  xts::optimize::trial_functions::QuadraticFunction<double> quadratic;

  xt::xarray<double> x = {1.0, 1.0};
  xt::xarray<double> direction = {-1.0, -1.0}; // Decreasing direction

  SECTION("Using WeakWolfeCondition with alpha_min = 0.2, alpha_max = 0.8") {
    xts::optimize::linesearch::conditions::WeakWolfeCondition<double> condition;
    xts::optimize::linesearch::search_strategy::BisectionSearch<double>
        bisection(condition, 0.2, 0.8);
    double alpha = bisection.search(quadratic, {x, direction});
    REQUIRE(condition(alpha, quadratic, {x, direction}) == true);
  }
}
TEST_CASE(
    "BisectionSearch Strategy Test with Different Tolerance and Max Iterations",
    "[BisectionSearch]") {
  xts::optimize::trial_functions::QuadraticFunction<double> quadratic;

  xt::xarray<double> x = {1.0, 1.0};
  xt::xarray<double> direction = {-1.0, -1.0};

  SECTION("Using WeakWolfeCondition with tol = 1e-3 and max_iterations = 5") {
    xts::optimize::OptimizeControl<double> control(5, 1e-3, false);
    xts::optimize::linesearch::conditions::WeakWolfeCondition<double> condition;
    xts::optimize::linesearch::search_strategy::BisectionSearch<double>
        bisection(condition, 0.0, 1.0, control);
    double alpha = bisection.search(quadratic, {x, direction});

    REQUIRE(condition(alpha, quadratic, {x, direction}) == true);
    REQUIRE(quadratic(x + alpha * direction) <
            quadratic(x)); // Test if the function value has decreased
  }

  // Add other sections with different tol and max_iterations values
}

TEST_CASE("BisectionSearch Convergence Test", "[BisectionSearch]") {
  xts::optimize::trial_functions::QuadraticFunction<double> quadratic;

  xt::xarray<double> x = {1.0, 1.0};
  xt::xarray<double> direction = {-1.0, -1.0};

  SECTION("Using WeakWolfeCondition") {
    xts::optimize::linesearch::conditions::WeakWolfeCondition<double> condition;
    xts::optimize::linesearch::search_strategy::BisectionSearch<double>
        bisection(condition);
    double alpha = bisection.search(quadratic, {x, direction});

    // Check if found alpha satisfies the condition
    REQUIRE(condition(alpha, quadratic, {x, direction}) == true);

    // Test if the function value at the new point is less than the value at the
    // starting point
    REQUIRE(quadratic(x + alpha * direction) < quadratic(x));
  }
}
