// MIT License
// Copyright 2023--present Rohit Goswami <HaoZeke>
#include "xtensor-io/xnpz.hpp"
#include "xtensor/xarray.hpp"

#include "include/xtensor_fmt.hpp"
#include "xtsci/optimize/linesearch/conditions/wolfe.hpp"
#include "xtsci/optimize/trial_functions/quadratic.hpp"

#include <catch2/catch_all.hpp>

TEST_CASE("WeakWolfeCondition Test", "[WeakWolfeCondition]") {
    xts::optimize::trial_functions::QuadraticFunction<double> quadratic;

    xt::xarray<double> x = {1.0, 1.0};
    xt::xarray<double> direction = {-1.0, -1.0};  // Decreasing direction

    SECTION("Should satisfy conditions for small step size") {
        xts::optimize::linesearch::conditions::WeakWolfeCondition<double> condition;
        REQUIRE(condition(0.1, quadratic, { x, direction }) == true);
    }

    SECTION("Should not satisfy conditions for large step size") {
        xts::optimize::linesearch::conditions::WeakWolfeCondition<double> condition;
        REQUIRE(condition(2.0, quadratic, { x, direction }) == false);
    }
}

TEST_CASE("StrongWolfeCondition Test", "[StrongWolfeCondition]") {
    xts::optimize::trial_functions::QuadraticFunction<double> quadratic;

    xt::xarray<double> x = {1.0, 1.0};
    xt::xarray<double> direction = {-1.0, -1.0};  // Decreasing direction

    SECTION("Should satisfy conditions for small step size") {
        xts::optimize::linesearch::conditions::StrongWolfeCondition<double> condition;
        REQUIRE(condition(0.1, quadratic, { x, direction }) == true);
    }

    SECTION("Should not satisfy conditions for large step size") {
        xts::optimize::linesearch::conditions::StrongWolfeCondition<double> condition;
        REQUIRE(condition(2.0, quadratic, { x, direction }) == false);
    }
}
