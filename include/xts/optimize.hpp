#pragma once

/**
 * \file xts/optimize.hpp
 * \brief C++ API for the Rust xtsci-optimize hourglass.
 *
 * Solvers live in Rust. This header wraps xts_minimize over dlpk tensors.
 * It does not reimplement BFGS, L-BFGS, NLCG, or line search in C++.
 * xtensor callers also include xts/xtensor.hpp.
 */

#include "../xts.h"

#include <cstddef>
#include <stdexcept>
#include <string>

namespace xts {
namespace optimize {

inline const char* version() noexcept { return xts_version(); }

enum class Method {
    PolakRibiere = XTS_POLAK_RIBIERE,
    FletcherReeves = XTS_FLETCHER_REEVES,
    Bfgs = XTS_BFGS,
    Lbfgs = XTS_LBFGS,
    Sr1 = XTS_SR1,
    Adam = XTS_ADAM,
    Steepest = XTS_STEEPEST,
    Sr2 = XTS_SR2,
    Pso = XTS_PSO,
    HestenesStiefel = XTS_HESTENES_STIEFEL,
    DaiYuan = XTS_DAI_YUAN,
    ConjugateDescent = XTS_CONJUGATE_DESCENT,
    HagerZhang = XTS_HAGER_ZHANG,
    LiuStorey = XTS_LIU_STOREY,
    FrPr = XTS_FR_PR,
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

using ScalarType = double;

struct OptimizeControl {
    std::size_t max_iterations = 100;
    ScalarType gtol = 1e-8;
    ScalarType tol = 1e-8;
    ScalarType istep = 0.1;
    std::size_t memory = 10;
    bool verbose = false;
    ScalarType maxmove = 1000;
    ScalarType xtol = 1e-6;
    ScalarType ftol = 1e-6;

    OptimizeControl() = default;
    OptimizeControl(std::size_t miter_val, ScalarType tol_val, bool verb_val)
        : max_iterations{miter_val}, gtol{tol_val}, tol{tol_val}, verbose{verb_val} {}

    Control to_control() const {
        return Control{max_iterations, gtol, istep, memory};
    }
};

struct OptimizeResult {
    ScalarType fun = 0.0;
    std::size_t nit = 0;
    ScalarType grad_norm = 0.0;
    bool success = true;
    int status = 0;

    static OptimizeResult from_report(Report const& r) {
        OptimizeResult out;
        out.fun = r.value;
        out.nit = r.steps;
        out.grad_norm = r.grad_norm;
        return out;
    }
};

inline Report minimize_fn(xts_eval_fn eval, xts_grad_fn grad, void* user,
                          DLManagedTensorVersioned* x, Control const& ctrl,
                          Method method) {
    xts_control_t c{ctrl.maxiter, ctrl.gtol, ctrl.istep, ctrl.memory};
    xts_report_t out{};
    xts_status_t st =
        xts_minimize(eval, grad, user, x, &c, static_cast<xts_method_t>(method), &out);
    if (st != XTS_SUCCESS) {
        char const* msg = xts_last_error();
        throw std::runtime_error(msg ? msg : "xts_minimize failed");
    }
    return Report{out.value, out.steps, out.grad_norm};
}

inline Report minimize_eindir(const eindir_objective_t* objective,
                              const eindir_abi_stamp_t* stamp,
                              DLManagedTensorVersioned* x, Control const& ctrl,
                              Method method) {
    xts_control_t c{ctrl.maxiter, ctrl.gtol, ctrl.istep, ctrl.memory};
    xts_report_t out{};
    xts_status_t st = xts_minimize_eindir(
        objective, stamp, x, &c, static_cast<xts_method_t>(method), &out);
    if (st != XTS_SUCCESS) {
        char const* msg = xts_last_error();
        throw std::runtime_error(msg ? msg : "xts_minimize_eindir failed");
    }
    return Report{out.value, out.steps, out.grad_norm};
}

inline OptimizeResult minimize(xts_eval_fn eval, xts_grad_fn grad, void* user,
                               DLManagedTensorVersioned* x,
                               OptimizeControl const& ctrl, Method method) {
    return OptimizeResult::from_report(
        minimize_fn(eval, grad, user, x, ctrl.to_control(), method));
}

inline DLManagedTensorVersioned* borrow_cpu_f64(double* data, std::size_t n) {
    return xts_tensor_borrow_cpu_f64(data, n);
}

namespace minimize {

inline constexpr Method BFGSOptimizer = Method::Bfgs;
inline constexpr Method LBFGSOptimizer = Method::Lbfgs;
inline constexpr Method SR1Optimizer = Method::Sr1;
inline constexpr Method SR2Optimizer = Method::Sr2;
inline constexpr Method ADAMOptimizer = Method::Adam;
inline constexpr Method SteepestDescentOptimizer = Method::Steepest;
inline constexpr Method ConjugateGradientOptimizer = Method::PolakRibiere;
inline constexpr Method PSOptim = Method::Pso;

}  // namespace minimize

namespace nlcg {
namespace conjugacy {

inline constexpr Method PolakRibiere = Method::PolakRibiere;
inline constexpr Method FletcherReeves = Method::FletcherReeves;
inline constexpr Method HestenesStiefel = Method::HestenesStiefel;
inline constexpr Method DaiYuan = Method::DaiYuan;
inline constexpr Method ConjugateDescent = Method::ConjugateDescent;
inline constexpr Method HagerZhang = Method::HagerZhang;
inline constexpr Method LiuStorey = Method::LiuStorey;
inline constexpr Method FrPr = Method::FrPr;

}  // namespace conjugacy
}  // namespace nlcg

}  // namespace optimize
}  // namespace xts
