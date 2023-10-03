#pragma once
// MIT License
// Copyright 2023--present Rohit Goswami <HaoZeke>
#include "fmt/core.h"
#include "fmt/format.h"
#include "xtensor/xarray.hpp"

namespace fmt {
template <typename T> struct formatter<xt::xarray<T>> {
  // Parse format specifiers if needed. Empty for demonstration.
  template <typename ParseContext> constexpr auto parse(ParseContext &ctx) {
    return ctx.begin();
  }

  // Format the xarray object and append it to the output.
  template <typename FormatContext>
  auto format(const xt::xarray<T> &arr, FormatContext &ctx) {
    auto out = ctx.out();
    if (arr.dimension() == 1) {
      // Handle 1D array
      out = fmt::format_to(out, "[");
      for (std::size_t i = 0; i < arr.shape()[0]; ++i) {
        if (i > 0)
          out = fmt::format_to(out, ", ");
        out = fmt::format_to(out, "{}", arr(i));
      }
      out = fmt::format_to(out, "]");
    } else if (arr.dimension() == 2) {
      // Handle 2D array
      out = fmt::format_to(out, "[\n");
      for (std::size_t i = 0; i < arr.shape()[0]; ++i) {
        out = fmt::format_to(out, "  [");
        for (std::size_t j = 0; j < arr.shape()[1]; ++j) {
          if (j > 0)
            out = fmt::format_to(out, ", ");
          out = fmt::format_to(out, "{}", arr(i, j));
        }
        out = fmt::format_to(out, "]");
        if (i < arr.shape()[0] - 1) {
          out = fmt::format_to(out, ",");
        }
        out = fmt::format_to(out, "\n");
      }
      out = fmt::format_to(out, "]");
    } else {
      // Unsupported dimensions
      out = fmt::format_to(out, "Only 1D and 2D arrays are supported.");
    }
    return out;
  }
};
} // namespace fmt
