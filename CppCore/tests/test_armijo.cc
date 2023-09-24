// MIT License
// Copyright 2023--present Rohit Goswami <HaoZeke>
#include "xtensor-io/xnpz.hpp"
#include "xtensor/xarray.hpp"

#include "include/xtensor_fmt.hpp"
#include "xtsci/optimize/linesearch/conditions/armijo.hpp"
#include "xtsci/optimize/trial_functions/quadratic.hpp"

#include <catch2/catch_all.hpp>

TEST_CASE("Armijo condition is validated", "[LineSearch]") {
  using Scalar = double;
  xts::optimize::linesearch::conditions::ArmijoCondition<Scalar>
      armijo_condition(0.1); // c=0.1 for demonstration
  xts::optimize::trial_functions::QuadraticFunction<Scalar> func;

  xt::xarray<Scalar> x = xt::xarray<Scalar>::from_shape({2});
  x.fill(1.0);                                 // start at point [1, 1]
  xt::xarray<Scalar> direction = {-1.0, -1.0}; // move in the [-1, -1] direction

  SECTION("Condition holds for valid step size") {
    Scalar alpha = 0.1;
    REQUIRE(armijo_condition(alpha, func, {x, direction}) == true);
  }

  SECTION("Condition does not hold for large step size") {
    Scalar alpha = 2.0;
    REQUIRE(armijo_condition(alpha, func, {x, direction}) == false);
  }

  SECTION("Condition holds at boundary") {
    Scalar alpha = 1.0; // This is just at the boundary
    REQUIRE(armijo_condition(alpha, func, {x, direction}) == true);
  }
}
