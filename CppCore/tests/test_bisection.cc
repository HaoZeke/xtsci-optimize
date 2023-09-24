// MIT License
// Copyright 2023--present Rohit Goswami <HaoZeke>
#include "xtensor-io/xnpz.hpp"
#include "xtensor/xarray.hpp"

#include "include/xtensor_fmt.hpp"
#include "xtsci/optimize/linesearch/search_strategy/bisection.hpp"
#include "xtsci/optimize/trial_functions/quadratic.hpp"
#include "xtsci/optimize/linesearch/conditions/wolfe.hpp"
#include "xtsci/optimize/linesearch/conditions/armijo.hpp"
#include "xtsci/optimize/linesearch/conditions/curvature.hpp"

#include <catch2/catch_all.hpp>

TEST_CASE("BisectionSearch Strategy Test", "[BisectionSearch]") {
    xts::optimize::trial_functions::QuadraticFunction<double> quadratic;

    xt::xarray<double> x = {1.0, 1.0};
    xt::xarray<double> direction = {-1.0, -1.0};  // Decreasing direction

    xts::optimize::linesearch::search_strategy::BisectionSearch<double> bisection;

    SECTION("Using WeakWolfeCondition") {
        xts::optimize::linesearch::conditions::WeakWolfeCondition<double> condition;
        double alpha = bisection.search(quadratic, { x, direction }, condition);
        // Let's check if the found alpha satisfies the condition
        REQUIRE(condition(alpha, quadratic, { x, direction }) == true);
    }

    SECTION("Using StrongWolfeCondition") {
        xts::optimize::linesearch::conditions::StrongWolfeCondition<double> condition;
        double alpha = bisection.search(quadratic, { x, direction }, condition);
        // Let's check if the found alpha satisfies the condition
        REQUIRE(condition(alpha, quadratic, { x, direction }) == true);
    }

    SECTION("Using ArmijoCondition") {
        xts::optimize::linesearch::conditions::ArmijoCondition<double> condition;
        double alpha = bisection.search(quadratic, { x, direction }, condition);
        // Let's check if the found alpha satisfies the condition
        REQUIRE(condition(alpha, quadratic, { x, direction }) == true);
    }

    SECTION("Using CurvatureCondition") {
        xts::optimize::linesearch::conditions::CurvatureCondition<double> condition;
        double alpha = bisection.search(quadratic, { x, direction }, condition);
        // Let's check if the found alpha satisfies the condition
        REQUIRE(condition(alpha, quadratic, { x, direction }) == true);
    }

    SECTION("Using StrongCurvatureCondition") {
        xts::optimize::linesearch::conditions::StrongCurvatureCondition<double> condition;
        double alpha = bisection.search(quadratic, { x, direction }, condition);
        // Let's check if the found alpha satisfies the condition
        REQUIRE(condition(alpha, quadratic, { x, direction }) == true);
    }
}
