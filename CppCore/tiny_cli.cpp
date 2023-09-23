// MIT License
// Copyright 2023--present Rohit Goswami <HaoZeke>
#include <fmt/format.h>
#include <fmt/ostream.h>

#include <cmath>
#include <cstdlib>

#include <iostream>

#include "include/xtensor_fmt.hpp"
#include "include/xtensor_trial_funcs.hpp"
#include "xtensor-io/xnpz.hpp"
#include "xtensor/xarray.hpp"

using namespace xts::funcs::trial;
int main(int argc, char *argv[]) {
  // Eat warnings, also safer
  static_cast<void>(argc);
  static_cast<void>(argv);
  // Define the grid
  size_t n_points = 100; // Number of points along each axis
  double x_min = -2.0, x_max = 2.0;
  double y_min = -2.0, y_max = 2.0;

    // Generate grid
    xt::xtensor<double, 1> x = xt::linspace<double>(-2.0, 2.0, 100);
    xt::xtensor<double, 1> y = xt::linspace<double>(-2.0, 2.0, 100);

    xt::xtensor<double, 2> X, Y;
    std::tie(X, Y) = xt::meshgrid(x, y);

    // Evaluate Rosenbrock function on the grid
    xt::xtensor<double, 2> Z = rosenbrock<double>(X, Y);

    // Save to disk
    // Write data to NPZ
    xt::dump_npz("rosenbrock.npz", "X", X);
    xt::dump_npz("rosenbrock.npz", "Y", Y, true); // The 'true' means append to existing file
    xt::dump_npz("rosenbrock.npz", "Z", Z, true);

    std::cout << "Data written to rosenbrock.npz" << std::endl;
  return EXIT_SUCCESS;
}
