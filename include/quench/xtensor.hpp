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
    auto as_xarray = [](DLManagedTensorVersioned const* t) {
        auto const& dl = t->dl_tensor;
        auto n_ = static_cast<std::size_t>(dl.shape[0]);
        auto* p = reinterpret_cast<double const*>(
            static_cast<unsigned char const*>(dl.data) + dl.byte_offset);
        ::xt::xarray<double> xv = ::xt::zeros<double>({n_});
        std::copy(p, p + n_, xv.begin());
        return xv;
    };
    auto eval_c = [](void* user, DLManagedTensorVersioned const* t,
                     double* value_out) -> quench_status_t {
        auto* b = static_cast<Box*>(user);
        *value_out = b->eval(as_xarray(t));
        return QUENCH_SUCCESS;
    };
    auto grad_c = [](void* user, DLManagedTensorVersioned const* t,
                     DLManagedTensorVersioned* g) -> quench_status_t {
        auto* b = static_cast<Box*>(user);
        auto gv = b->grad(as_xarray(t));
        auto const& dl = g->dl_tensor;
        auto n_ = static_cast<std::size_t>(dl.shape[0]);
        auto* p = reinterpret_cast<double*>(
            static_cast<unsigned char*>(dl.data) + dl.byte_offset);
        std::copy(gv.begin(), gv.end(), p);
        (void)n_;
        return QUENCH_SUCCESS;
    };
    std::vector<double> buf(x.begin(), x.end());
    DLManagedTensorVersioned* xt = borrow_cpu_f64(buf.data(), n);
    Report r = minimize_fn(eval_c, grad_c, &box, xt, ctrl, method);
    quench_tensor_free(xt);
    std::copy(buf.begin(), buf.end(), x.begin());
    return r;
}

}  // namespace xt
}  // namespace quench
