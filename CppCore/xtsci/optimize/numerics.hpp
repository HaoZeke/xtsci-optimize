#pragma once
// MIT License
// Copyright 2023--present Rohit Goswami <HaoZeke>

#include "xtsci/func/base.hpp"

namespace xts::optimize {
using ScalarType = double;
using BoolVec = xt::xtensor<bool, 1>;
using ScalarVec = xt::xtensor<ScalarType, 1, xt::layout_type::row_major>;
using ScalarMatrix = xt::xtensor<ScalarType, 2, xt::layout_type::row_major>;
using FObjFunc = func::ObjectiveFunction<ScalarType>;
} // namespace xts::optimize
