#pragma once

/**
 * \file quench/xtensor.hpp
 * \brief xt::xarray adapter onto the quench C ABI.
 *
 * The xtensor types never enter Rust. This header copies a 1-D xarray
 * to a contiguous buffer, runs quench_minimize_fn, and writes back.
 */

#include "../quench.hpp"

#include <xtensor/xarray.hpp>
#include <xtensor/xmath.hpp>

namespace quench {
namespace xt {

template <class E, class G>
inline Report minimize(E&& eval, G&& grad, ::xt::xarray<double>& x,
                       Control const& ctrl, Method method) {
    if (x.dimension() != 1) {
        throw std::runtime_error("quench::xt::minimize expects a 1-D xarray");
    }
    auto n = static_cast<std::size_t>(x.size());
    struct Box {
        E eval;
        G grad;
    };
    Box box{std::forward<E>(eval), std::forward<G>(grad)};
    auto eval_c = [](double const* p, std::size_t n_, void* user) -> double {
        auto* b = static_cast<Box*>(user);
        ::xt::xarray<double> xv = ::xt::zeros<double>({n_});
        std::copy(p, p + n_, xv.begin());
        return b->eval(xv);
    };
    auto grad_c = [](double const* p, double* g, std::size_t n_, void* user) {
        auto* b = static_cast<Box*>(user);
        ::xt::xarray<double> xv = ::xt::zeros<double>({n_});
        std::copy(p, p + n_, xv.begin());
        auto gv = b->grad(xv);
        std::copy(gv.begin(), gv.end(), g);
    };
    std::vector<double> buf(x.begin(), x.end());
    Report r = minimize_fn(eval_c, grad_c, &box, buf.data(), n, ctrl, method);
    std::copy(buf.begin(), buf.end(), x.begin());
    return r;
}

}  // namespace xt
}  // namespace quench
