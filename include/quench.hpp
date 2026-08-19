#pragma once

/**
 * \file quench.hpp
 * \brief C++ hourglass over the quench C ABI.
 *
 * Rust owns the algorithms. This header is a thin inline wrapper, the same
 * shape as metatensor.hpp / eindir-core.hpp: no second implementation.
 */

#include "quench.h"

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace quench {

inline const char* version() noexcept { return quench_version(); }

enum class Method {
    PolakRibiere = QUENCH_POLAK_RIBIERE,
    FletcherReeves = QUENCH_FLETCHER_REEVES,
    Bfgs = QUENCH_BFGS,
    Lbfgs = QUENCH_LBFGS,
    Sr1 = QUENCH_SR1,
    Adam = QUENCH_ADAM,
    Steepest = QUENCH_STEEPEST,
    Sr2 = QUENCH_SR2,
    Pso = QUENCH_PSO,
    HestenesStiefel = QUENCH_HESTENES_STIEFEL,
    DaiYuan = QUENCH_DAI_YUAN,
    ConjugateDescent = QUENCH_CONJUGATE_DESCENT,
    HagerZhang = QUENCH_HAGER_ZHANG,
    LiuStorey = QUENCH_LIU_STOREY,
    FrPr = QUENCH_FR_PR,
};

struct Control {
    std::size_t maxiter = 100;
    double gtol = 1e-8;
    double istep = 0.1;
    std::size_t memory = 10;
};

struct Report {
    double value = 0.0;
    std::size_t steps = 0;
    double grad_norm = 0.0;
};

inline Report minimize_fn(quench_eval_fn eval, quench_grad_fn grad, void* user,
                          DLManagedTensorVersioned* x, Control const& ctrl,
                          Method method) {
    quench_control_t c{ctrl.maxiter, ctrl.gtol, ctrl.istep, ctrl.memory};
    quench_report_t out{};
    quench_status_t st = quench_minimize_fn(
        eval, grad, user, x, &c, static_cast<quench_method_t>(method), &out);
    if (st == QUENCH_UNSUPPORTED_DEVICE) {
        throw std::runtime_error(quench_last_error());
    }
    if (st != QUENCH_SUCCESS) {
        char const* msg = quench_last_error();
        throw std::runtime_error(msg ? msg : "quench_minimize_fn failed");
    }
    return Report{out.value, out.steps, out.grad_norm};
}

/// Borrow `x[0..n]` as DLPack, run the hourglass, write the accepted point back.
inline Report minimize_fn(quench_eval_fn eval, quench_grad_fn grad, void* user,
                          double* x, std::size_t n, Control const& ctrl,
                          Method method) {
    if (x == nullptr || n == 0) {
        throw std::runtime_error("quench::minimize_fn: null or empty x");
    }
    DLManagedTensorVersioned* xt = quench_tensor_borrow_cpu_f64(x, n);
    if (xt == nullptr) {
        char const* msg = quench_last_error();
        throw std::runtime_error(msg ? msg : "quench_tensor_borrow_cpu_f64 failed");
    }
    try {
        Report r = minimize_fn(eval, grad, user, xt, ctrl, method);
        quench_tensor_free(xt);
        return r;
    } catch (...) {
        quench_tensor_free(xt);
        throw;
    }
}

/// Borrow a host f64 buffer as DLPack. Pair with quench_tensor_free.
inline DLManagedTensorVersioned* borrow_cpu_f64(double* data, std::size_t n) {
    return quench_tensor_borrow_cpu_f64(data, n);
}

}  // namespace quench
