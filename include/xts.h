#ifndef XTS_OPTIMIZE_H
#define XTS_OPTIMIZE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>
#include <dlpack/dlpack.h>

/* The direct eindir entry point only needs opaque handles here. Consumers
 * that use eindir_core's constructors can include its full header first. */
#if defined(__has_include)
#  if __has_include(<eindir-core.h>)
#    include <eindir-core.h>
#  else
typedef struct eindir_objective_t eindir_objective_t;
typedef struct eindir_abi_stamp_t eindir_abi_stamp_t;
#  endif
#else
#  include <eindir-core.h>
#endif

/** \file xts.h
 *  \brief C ABI for the Rust xtsci-optimize hourglass.
 *
 *  Solvers live in Rust. This header is the only C entry:
 *  \ref xts_minimize over dlpk \c DLManagedTensorVersioned tensors.
 */

/** Status of an ABI call. */
typedef enum xts_status_t {
    XTS_SUCCESS = 0,
    XTS_INVALID_PARAMETER = 1,
    XTS_INTERNAL_ERROR = 2,
    /** Tensor device is not CPU. The ABI stays stable for a later CUDA path. */
    XTS_UNSUPPORTED_DEVICE = 3
} xts_status_t;

/** Compatibility identity for the xtsci-optimize C ABI. */
typedef struct xts_abi_stamp_t {
    uint16_t abi_major;
    uint16_t abi_minor;
    uint16_t layout_revision;
} xts_abi_stamp_t;

#define XTS_ABI_VERSION_MAJOR 1
#define XTS_ABI_VERSION_MINOR 8
#define XTS_ABI_LAYOUT_REVISION 2

/** Solver selector. \c XTS_LBFGS is the production unconstrained method. */
typedef enum xts_method_t {
    XTS_POLAK_RIBIERE = 0,
    XTS_FLETCHER_REEVES = 1,
    XTS_BFGS = 2,
    XTS_LBFGS = 3,
    XTS_SR1 = 4,
    XTS_ADAM = 5,
    XTS_STEEPEST = 6,
    XTS_SR2 = 7,
    XTS_PSO = 8,
    XTS_HESTENES_STIEFEL = 9,
    XTS_DAI_YUAN = 10,
    XTS_CONJUGATE_DESCENT = 11,
    XTS_HAGER_ZHANG = 12,
    XTS_LIU_STOREY = 13,
    XTS_FR_PR = 14,
    XTS_NEWTON = 15,
    XTS_RFO = 16,
    XTS_FIRE = 17,
    XTS_BB = 18,
    XTS_DOGLEG = 19,
    XTS_FIRE2 = 20
} xts_method_t;

/** Outer-loop controls. \c memory is the L-BFGS pair cap. */
typedef struct xts_control_t {
    size_t maxiter;
    double gtol;
    double istep;
    size_t memory;
    double maxmove;
} xts_control_t;

/** Result of \ref xts_minimize. */
typedef struct xts_report_t {
    double value;
    size_t steps;
    double grad_norm;
} xts_report_t;

typedef xts_status_t (*xts_eval_fn)(void *user, const DLManagedTensorVersioned *x,
                                    double *value_out);
typedef xts_status_t (*xts_grad_fn)(void *user, const DLManagedTensorVersioned *x,
                                    DLManagedTensorVersioned *grad_out);
typedef xts_status_t (*xts_hess_fn)(void *user, const DLManagedTensorVersioned *x,
                                    DLManagedTensorVersioned *hess_out);
/** Fused (f, grad). One geometry, one host potential call. */
typedef xts_status_t (*xts_evalgrad_fn)(void *user,
                                        const DLManagedTensorVersioned *x,
                                        double *value_out,
                                        DLManagedTensorVersioned *grad_out);

/** Crate version string. */
const char *xts_version(void);
/** Return the C ABI compatibility identity for this build. */
xts_abi_stamp_t xts_abi_stamp(void);
/** Return nonzero when a stamp is compatible with this build. */
int32_t xts_abi_compatible(const xts_abi_stamp_t *stamp);
/** Thread-local last error message after a non-success status. */
const char *xts_last_error(void);
/** Borrow a CPU f64 buffer as a dlpk tensor. */
DLManagedTensorVersioned *xts_tensor_borrow_cpu_f64(double *data, size_t n);
/** Free a tensor allocated by this ABI. */
void xts_tensor_free(DLManagedTensorVersioned *tensor);
/**
 * Minimize from \a x in place.
 *
 * \param eval  Value callback.
 * \param grad  Gradient callback.
 * \param user  Passed through to both callbacks.
 * \param x     CPU f64 state (updated on success).
 * \param ctrl  Iteration / L-BFGS memory controls.
 * \param method Solver. \c XTS_LBFGS is the production choice.
 * \param out   Filled on success.
 */
xts_status_t xts_minimize(xts_eval_fn eval, xts_grad_fn grad, void *user,
                          DLManagedTensorVersioned *x, const xts_control_t *ctrl,
                          xts_method_t method, xts_report_t *out);
/**
 * Newton / RFO. \a hess writes a length-\c n*n row-major Hessian.
 * \c method is \c XTS_NEWTON or \c XTS_RFO.
 */
xts_status_t xts_minimize_hess(xts_eval_fn eval, xts_grad_fn grad,
                               xts_hess_fn hess, void *user,
                               DLManagedTensorVersioned *x,
                               const xts_control_t *ctrl, xts_method_t method,
                               xts_report_t *out);

/**
 * Minimize an eindir-compatible objective without taking ownership of it.
 * The stamp must be compatible with this build and include an analytic
 * gradient. The caller retains ownership of an objective.
 */
xts_status_t xts_minimize_eindir(
    const eindir_objective_t *objective, const eindir_abi_stamp_t *stamp,
    DLManagedTensorVersioned *x, const xts_control_t *ctrl, xts_method_t method,
    xts_report_t *out);

/**
 * Opaque session. Algorithm memory (L-BFGS pairs, NLCG directions,
 * dense H, Adam moments, PSO swarm) lives here. \c x stays a dlpk
 * tensor. Callbacks are arguments of each step, not stored.
 */
typedef struct xts_solver_t xts_solver_t;

/** Allocate a session. \a dim is the length of \c x. Null on error. */
xts_solver_t *xts_solver_create(xts_method_t method, const xts_control_t *ctrl,
                                size_t dim);
/** Release a session from \ref xts_solver_create. */
void xts_solver_free(xts_solver_t *solver);
/** Drop method memory. The next step is a cold start from the current \c x. */
void xts_solver_forget(xts_solver_t *solver);
/** Euclidean step cap for the next \ref xts_solver_step (saddle \c max_move). */
void xts_solver_set_maxmove(xts_solver_t *solver, double maxmove);
/** How an L-BFGS session uses a caller Hessian (eOn \c lbfgs_step). */
typedef enum xts_qn_step_t {
    XTS_QN_LBFGS = 0,
    XTS_QN_NEWTON = 1,
    XTS_QN_RFO = 2
} xts_qn_step_t;
/** Two-loop + H0, or Newton/RFO on P. Legal with \ref xts_solver_step_hess. */
void xts_solver_set_qn_step(xts_solver_t *solver, xts_qn_step_t step);
/** How a session takes a proposed step (eOn lbfgs_accept). */
typedef enum xts_accept_t {
    XTS_ACCEPT_NONE = 0,
    XTS_ACCEPT_ENERGY = 1,
    XTS_ACCEPT_NONMONOTONE = 2
} xts_accept_t;
void xts_solver_set_accept(xts_solver_t *solver, xts_accept_t accept);
void xts_solver_set_atom_maxmove(xts_solver_t *solver, double maxmove);
void xts_solver_set_project_rigid(xts_solver_t *solver, int32_t enabled);
void xts_solver_set_extra_updates(xts_solver_t *solver, size_t extra);
void xts_solver_set_cautious(xts_solver_t *solver, double eps, double alpha);
/** HiGHS feasible-set step. Nonzero enables it. Returns 0, or 1 if this
 *  build has no highs feature. */
int32_t xts_solver_set_highs(xts_solver_t *solver, int32_t enabled);
/** Embedded manifold. Euclidean is the default. */
typedef enum xts_manifold_t {
    XTS_MANIFOLD_EUCLIDEAN = 0,
    XTS_MANIFOLD_SPHERE = 1,
    XTS_MANIFOLD_SO3 = 2,
    XTS_MANIFOLD_STIEFEL = 3,
    XTS_MANIFOLD_SE3 = 4
} xts_manifold_t;
void xts_solver_set_manifold(xts_solver_t *solver, xts_manifold_t manifold);
/**
 * One outer iteration: direction, line search, curvature update.
 * \a eval and \a grad are valid for this call only. \a x is in/out.
 */
xts_status_t xts_solver_step(xts_solver_t *solver, xts_eval_fn eval,
                             xts_grad_fn grad, void *user,
                             DLManagedTensorVersioned *x, xts_report_t *out);
/** One Newton / RFO iteration. \a hess writes a length-\c n*n Hessian. */
xts_status_t xts_solver_step_hess(xts_solver_t *solver, xts_eval_fn eval,
                                  xts_grad_fn grad, xts_hess_fn hess,
                                  void *user, DLManagedTensorVersioned *x,
                                  xts_report_t *out);
/** Like \ref xts_solver_step with one fused (f, g) callback. */
xts_status_t xts_solver_step_fg(xts_solver_t *solver, xts_evalgrad_fn evalgrad,
                                void *user, DLManagedTensorVersioned *x,
                                xts_report_t *out);
/** Like \ref xts_solver_step_hess with one fused (f, g) callback. */
xts_status_t xts_solver_step_hess_fg(xts_solver_t *solver,
                                     xts_evalgrad_fn evalgrad, xts_hess_fn hess,
                                     void *user, DLManagedTensorVersioned *x,
                                     xts_report_t *out);

#ifdef __cplusplus
}
#endif

#endif /* XTS_OPTIMIZE_H */
