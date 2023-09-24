// MIT License
// Copyright 2023--present Rohit Goswami <HaoZeke>
#include "xtensor-io/xnpz.hpp"
#include "xtensor/xarray.hpp"

#include "include/xtensor_fmt.hpp"
#include "xtsci/optimize/linesearch/conditions/curvature.hpp"

#include <catch2/catch_all.hpp>
// Mock objective function for testing purposes
template <typename ScalarType>
class QuadraticFunction : public xts::optimize::ObjectiveFunction<ScalarType> {
public:
  ScalarType operator()(const xt::xarray<ScalarType> &x) const override {
    return xt::linalg::dot(x, x)(0); // x^T x
  }

  std::optional<xt::xarray<ScalarType>>
  gradient(const xt::xarray<ScalarType> &x) const override {
    return 2.0 * x; // 2x
  }
};

TEST_CASE("Curvature condition is validated", "[LineSearch]") {
  using Scalar = double;
  xts::optimize::linesearch::conditions::CurvatureCondition<Scalar>
      curvature_condition(0.9); // c'=0.9 for demonstration
  QuadraticFunction<Scalar> func;

  xt::xarray<Scalar> x = xt::xarray<Scalar>::from_shape({2});
  x.fill(1.0);                                 // start at point [1, 1]
  xt::xarray<Scalar> direction = {-1.0, -1.0}; // move in the [-1, -1] direction

  SECTION("Condition holds for valid step size") {
    Scalar alpha = 0.5;
    REQUIRE(curvature_condition(alpha, func, x, direction) == true);
  }

  SECTION("Condition does not hold for small step size") {
    Scalar alpha = 0.01;
    REQUIRE(curvature_condition(alpha, func, x, direction) == false);
  }

  SECTION("Condition holds at boundary") {
    Scalar alpha =
        1.0; // This should be just at the boundary for the quadratic function
    REQUIRE(curvature_condition(alpha, func, x, direction) == true);
  }
}
