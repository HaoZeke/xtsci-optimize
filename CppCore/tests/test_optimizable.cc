#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "xtsci/func/trial/D2/mullerbrown.hpp"

#include "xtsci/optimize/numerics.hpp"
#include "xtsci/optimize/optimizable.hpp"

using namespace xts::optimize;
constexpr double TEST_EPS{1e-4};

// Stub implementation for testing
class StubOptimizable : public xts::optimize::Optimizable {
public:
  StubOptimizable(FObjFunc &func) : Optimizable(func) {}

  ScalarType operator()(const ScalarVec &x) const override {
    // Simple quadratic function for testing
    return xt::linalg::dot(x, x)();
  }

  std::optional<ScalarVec> gradient(const ScalarVec &x) const override {
    // Gradient of the quadratic function
    return 2 * x;
  }

  ScalarVec diff(const ScalarVec &_a, const ScalarVec &_b) const override {
    return (_a - _b) / 2;
  }

  ScalarType directional_derivative(const ScalarVec &x,
                                    const ScalarVec &dir) const override {
    return static_cast<ScalarType>(14);
  }

  void setState(ScalarVec x) override { this->m_cstate = x; }
};

TEST_CASE("Optimizable Base Functionality", "[Optimizable]") {
  xts::func::trial::D2::MullerBrown<ScalarType> mullerBrown;
  StubOptimizable optimizable(
      mullerBrown); // StubOptimizable overrides this anyway

  SECTION("Function evaluation correctness") {
    xt::xarray<double> x = {1.0, 2.0, 3.0};
    double expectedValue = xt::linalg::dot(x, x)();
    REQUIRE_THAT(optimizable(x),
                 Catch::Matchers::WithinAbs(expectedValue, TEST_EPS));
  }

  SECTION("Gradient correctness") {
    xt::xarray<double> x = {1.0, 2.0, 3.0};
    auto grad = optimizable.gradient(x);
    xt::xarray<double> expectedGrad = 2 * x;
    REQUIRE(grad.has_value());
    REQUIRE(xt::all(xt::equal(*grad, expectedGrad)));
  }

  SECTION("Vector difference correctness") {
    xt::xarray<double> a = {5.0, 7.0};
    xt::xarray<double> b = {3.0, 2.0};
    auto diff = optimizable.diff(a, b);
    xt::xarray<double> expectedDiff = {1.0, 2.5};
    REQUIRE(xt::all(xt::equal(diff, expectedDiff)));
  }

  SECTION("State management") {
    xt::xarray<double> newState = {1.0, 2.0};
    optimizable.setState(newState);
    REQUIRE(xt::all(xt::equal(optimizable.getState(), newState)));
  }

  SECTION("Directional derivative") {
    xt::xarray<double> newState = {1.0, 2.0};
    ScalarType val = optimizable.directional_derivative(newState, newState);
    REQUIRE(val == 14);
  }
}

TEST_CASE("TOpt Functionality with Predefined MullerBrown", "[Optimizable]") {
  xts::func::trial::D2::MullerBrown<ScalarType> mullerBrown;
  xts::optimize::TOpt optimizable(mullerBrown);
  ScalarVec x{-0.558, 1.442};

  SECTION("Function evaluation") {
    REQUIRE_THAT(optimizable(x),
                 Catch::Matchers::WithinAbs(-146.69948920058778, TEST_EPS));
  }
  SECTION("Gradient calculation at known positions") {
    x = {1.623, 0.38};
    auto grad = optimizable.gradient(x);
    REQUIRE(grad.has_value());
    REQUIRE_THAT(grad.value()(0),
                 Catch::Matchers::WithinAbs(3075.34429488, TEST_EPS));
    REQUIRE_THAT(grad.value()(1),
                 Catch::Matchers::WithinAbs(873.2579683, TEST_EPS));
  }
  SECTION("Directional derivative positions") {
    x = {1.623, 0.38};
    ScalarVec direction{1, 2}, grad_at_x{3075.34429488, 873.2579683};
    ScalarType dirderv = optimizable.directional_derivative(x, direction);
    REQUIRE_THAT(optimizable.directional_derivative(x, direction),
                 Catch::Matchers::WithinAbs(
                     xt::linalg::dot(grad_at_x, direction)(), TEST_EPS));
  }
}
