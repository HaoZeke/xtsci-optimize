#pragma once
// MIT License
// Copyright 2023--present Rohit Goswami <HaoZeke>
#include "fmt/core.h"
#include "fmt/format.h"
#include "xtensor/xmath.hpp"
#include "xtensor/xtensor.hpp"

namespace xts {
namespace funcs {
namespace trial {

// Rosenbrock function definition for 2D tensors
template <typename T>
xt::xtensor<T, 2> rosenbrock(const xt::xtensor<T, 2>& x, const xt::xtensor<T, 2>& y) {
    return (1.0 - x) * (1.0 - x) + 100.0 * (y - x * x) * (y - x * x);
}

// Gradient of the Rosenbrock function for 2D tensors
template <typename T>
std::tuple<xt::xtensor<T, 2>, xt::xtensor<T, 2>> rosenbrock_gradient(const xt::xtensor<T, 2>& x, const xt::xtensor<T, 2>& y) {
    xt::xtensor<T, 2> dx = -2.0 * (1.0 - x) - 400.0 * x * (y - x * x);
    xt::xtensor<T, 2> dy = 200.0 * (y - x * x);

    return std::make_tuple(dx, dy);
}
} // namespace trial
} // namespace funcs
} // namespace xts
