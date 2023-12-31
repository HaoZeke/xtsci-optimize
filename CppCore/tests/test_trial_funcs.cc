// MIT License
// Copyright 2023--present Rohit Goswami <HaoZeke>
#include "xtensor-io/xnpz.hpp"
#include "xtensor/xarray.hpp"

#include "include/xtensor_fmt.hpp"

#include "xtsci/optimize/trial_functions/quadratic.hpp"
#include "xtsci/optimize/trial_functions/rosenbrock.hpp"

#include <catch2/catch_all.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

TEST_CASE("Rosenbrock Function Evaluation", "[Rosenbrock]") {
  xts::optimize::trial_functions::Rosenbrock<double> rosen;

  SECTION("Function value at (1, 1)") {
    xt::xarray<double> point = {1.0, 1.0};
    REQUIRE_THAT(rosen(point), Catch::Matchers::WithinAbs(0.0, 1e-6));
  }

  SECTION("Function value at (0, 0)") {
    xt::xarray<double> point = {0.0, 0.0};
    REQUIRE_THAT(rosen(point), Catch::Matchers::WithinAbs(1.0, 1e-6));
  }

  SECTION("Gradient at (1, 1)") {
    xt::xarray<double> point = {1.0, 1.0};
    auto grad = rosen.gradient(point).value();
    REQUIRE_THAT(grad(0), Catch::Matchers::WithinAbs(0.0, 1e-6));
    REQUIRE_THAT(grad(1), Catch::Matchers::WithinAbs(0.0, 1e-6));
  }

  SECTION("Gradient at (0, 0)") {
    xt::xarray<double> point = {0.0, 0.0};
    auto grad = rosen.gradient(point).value();
    REQUIRE_THAT(grad(0), Catch::Matchers::WithinAbs(-2.0, 1e-6));
    REQUIRE_THAT(grad(1), Catch::Matchers::WithinAbs(0.0, 1e-6));
  }
  SECTION("Hessian at (1, 1)") {
    xt::xarray<double> point = {1.0, 1.0};
    auto hess = rosen.hessian(point).value();
    REQUIRE_THAT(hess(0, 0), Catch::Matchers::WithinAbs(802.0, 1e-6));
    REQUIRE_THAT(hess(0, 1), Catch::Matchers::WithinAbs(-400.0, 1e-6));
    REQUIRE_THAT(hess(1, 0), Catch::Matchers::WithinAbs(-400.0, 1e-6));
    REQUIRE_THAT(hess(1, 1), Catch::Matchers::WithinAbs(200.0, 1e-6));
  }

  SECTION("Hessian at (0, 0)") {
    xt::xarray<double> point = {0.0, 0.0};
    auto hess = rosen.hessian(point).value();
    REQUIRE_THAT(hess(0, 0), Catch::Matchers::WithinAbs(2.0, 1e-6));
    REQUIRE_THAT(hess(0, 1), Catch::Matchers::WithinAbs(0.0, 1e-6));
    REQUIRE_THAT(hess(1, 0), Catch::Matchers::WithinAbs(0.0, 1e-6));
    REQUIRE_THAT(hess(1, 1), Catch::Matchers::WithinAbs(200.0, 1e-6));
  }
}

TEST_CASE("Quadratic Function Evaluation", "[Quadratic]") {
  xts::optimize::trial_functions::QuadraticFunction<double> quadratic;

  SECTION("Function value at (1, 1)") {
    xt::xarray<double> point = {1.0, 1.0};
    REQUIRE_THAT(quadratic(point),
                 Catch::Matchers::WithinAbs(2.0, 1e-6)); // 1^2 + 1^2 = 2
  }

  SECTION("Function value at (0, 0)") {
    xt::xarray<double> point = {0.0, 0.0};
    REQUIRE_THAT(quadratic(point),
                 Catch::Matchers::WithinAbs(0.0, 1e-6)); // 0^2 + 0^2 = 0
  }

  SECTION("Gradient at (1, 1)") {
    xt::xarray<double> point = {1.0, 1.0};
    auto grad = quadratic.gradient(point).value();
    REQUIRE_THAT(grad(0), Catch::Matchers::WithinAbs(2.0, 1e-6)); // 2(1) = 2
    REQUIRE_THAT(grad(1), Catch::Matchers::WithinAbs(2.0, 1e-6)); // 2(1) = 2
  }

  SECTION("Gradient at (0, 0)") {
    xt::xarray<double> point = {0.0, 0.0};
    auto grad = quadratic.gradient(point).value();
    REQUIRE_THAT(grad(0), Catch::Matchers::WithinAbs(0.0, 1e-6)); // 2(0) = 0
    REQUIRE_THAT(grad(1), Catch::Matchers::WithinAbs(0.0, 1e-6)); // 2(0) = 0
  }

  SECTION("Hessian at any point") {
    xt::xarray<double> point = {
        1.0, 1.0}; // Hessian is constant and doesn't depend on the point
    auto hess = quadratic.hessian(point).value();
    REQUIRE_THAT(hess(0, 0), Catch::Matchers::WithinAbs(2.0, 1e-6));
    REQUIRE_THAT(hess(0, 1), Catch::Matchers::WithinAbs(0.0, 1e-6));
    REQUIRE_THAT(hess(1, 0), Catch::Matchers::WithinAbs(0.0, 1e-6));
    REQUIRE_THAT(hess(1, 1), Catch::Matchers::WithinAbs(2.0, 1e-6));
  }
}
