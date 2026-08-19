#pragma once

/**
 * \file xts/optimize.hpp
 * \brief Source-compatible xts::optimize names over the quench hourglass.
 *
 * Solvers live in Rust and are reached only through quench.hpp /
 * quench_minimize_fn. This header does not reimplement BFGS, L-BFGS, SR1,
 * Adam, NLCG, or line search in C++, and it does not pull xtensor.
 *
 * xtensor callers include quench/xtensor.hpp and then use these aliases,
 * or call quench::xt::minimize directly.
 *
 * Name map (HaoZeke/xtsci-optimize CppCore/xtsci/optimize):
 *   OptimizeControl / OptimizeResult  <-  quench::Control / quench::Report
 *   minimize::BFGSOptimizer           <-  quench::Method::Bfgs
 *   minimize::LBFGSOptimizer          <-  quench::Method::Lbfgs
 *   minimize::SR1Optimizer            <-  quench::Method::Sr1
 *   minimize::SR2Optimizer            <-  quench::Method::Sr2
 *   minimize::ADAMOptimizer           <-  quench::Method::Adam
 *   minimize::SteepestDescentOptimizer<-  quench::Method::Steepest
 *   minimize::PSOptim                 <-  quench::Method::Pso
 *   minimize::ConjugateGradientOptimizer
 *                                     <-  quench::Method::PolakRibiere
 *   nlcg::conjugacy::PolakRibiere     <-  quench::Method::PolakRibiere
 *   nlcg::conjugacy::FletcherReeves   <-  quench::Method::FletcherReeves
 *   nlcg::conjugacy::HestenesStiefel  <-  quench::Method::HestenesStiefel
 *   nlcg::conjugacy::DaiYuan          <-  quench::Method::DaiYuan
 *   nlcg::conjugacy::ConjugateDescent <-  quench::Method::ConjugateDescent
 *   nlcg::conjugacy::HagerZhang       <-  quench::Method::HagerZhang
 *   nlcg::conjugacy::LiuStorey        <-  quench::Method::LiuStorey
 *   nlcg::conjugacy::FrPr             <-  quench::Method::FrPr
 */

#include "../quench.hpp"

namespace xts {
namespace optimize {

/// Scalar used by the C ABI (original numerics.hpp alias).
using ScalarType = double;

using Method = ::quench::Method;
using Control = ::quench::Control;
using Report = ::quench::Report;

inline const char* version() noexcept { return ::quench::version(); }

/**
 * xtsci OptimizeControl field names, converted to quench::Control at the
 * hourglass. Only max_iterations, gtol, istep, and memory cross the ABI.
 * verbose / maxmove / xtol / ftol / tol are accepted for source compatibility
 * and are not sent to Rust.
 */
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

    /// Original three-argument constructor (base.hpp).
    OptimizeControl(std::size_t miter_val, ScalarType tol_val, bool verb_val)
        : max_iterations{miter_val},
          gtol{tol_val},
          tol{tol_val},
          verbose{verb_val} {}

    ::quench::Control to_quench() const {
        return ::quench::Control{max_iterations, gtol, istep, memory};
    }
};

/**
 * xtsci OptimizeResult names over quench::Report.
 * fun <- value, nit <- steps. Tensor fields (x, jac, hess) stay on the
 * caller side; the hourglass writes the accepted point back into x[].
 */
struct OptimizeResult {
    ScalarType fun = 0.0;
    std::size_t nit = 0;
    ScalarType grad_norm = 0.0;
    bool success = true;
    int status = 0;

    static OptimizeResult from_quench(::quench::Report const& r) {
        OptimizeResult out;
        out.fun = r.value;
        out.nit = r.steps;
        out.grad_norm = r.grad_norm;
        out.success = true;
        out.status = 0;
        return out;
    }
};

/// Line-search bracket (original AlphaState). Unused by the hourglass.
struct AlphaState {
    ScalarType init = 1.0;
    ScalarType low = 0.0;
    ScalarType hi = 1.0;
};

inline OptimizeResult minimize(quench_eval_fn eval, quench_grad_fn grad,
                               void* user, double* x, std::size_t n,
                               OptimizeControl const& ctrl, Method method) {
    Report r = ::quench::minimize_fn(eval, grad, user, x, n, ctrl.to_quench(),
                                     method);
    return OptimizeResult::from_quench(r);
}

inline OptimizeResult minimize(quench_eval_fn eval, quench_grad_fn grad,
                               void* user, double* x, std::size_t n,
                               Control const& ctrl, Method method) {
    Report r = ::quench::minimize_fn(eval, grad, user, x, n, ctrl, method);
    return OptimizeResult::from_quench(r);
}

inline Report minimize_fn(quench_eval_fn eval, quench_grad_fn grad, void* user,
                          double* x, std::size_t n, Control const& ctrl,
                          Method method) {
    return ::quench::minimize_fn(eval, grad, user, x, n, ctrl, method);
}

/// Original optimizer class names as Method tags (no C++ solver bodies).
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
