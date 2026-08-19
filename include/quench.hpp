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
                          double* x, std::size_t n, Control const& ctrl,
                          Method method) {
    quench_control_t c{ctrl.maxiter, ctrl.gtol, ctrl.istep, ctrl.memory};
    quench_report_t out{};
    quench_status_t st = quench_minimize_fn(
        eval, grad, user, x, n, &c,
        static_cast<quench_method_t>(method), &out);
    if (st != QUENCH_SUCCESS) {
        char const* msg = quench_last_error();
        throw std::runtime_error(msg ? msg : "quench_minimize_fn failed");
    }
    return Report{out.value, out.steps, out.grad_norm};
}

}  // namespace quench
