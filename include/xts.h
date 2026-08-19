#ifndef XTS_OPTIMIZE_H
#define XTS_OPTIMIZE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>
#include <dlpack/dlpack.h>

typedef enum xts_status_t {
    XTS_SUCCESS = 0,
    XTS_INVALID_PARAMETER = 1,
    XTS_INTERNAL_ERROR = 2,
    XTS_UNSUPPORTED_DEVICE = 3
} xts_status_t;

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
    XTS_FR_PR = 14
} xts_method_t;

typedef struct xts_control_t {
    size_t maxiter;
    double gtol;
    double istep;
    size_t memory;
} xts_control_t;

typedef struct xts_report_t {
    double value;
    size_t steps;
    double grad_norm;
} xts_report_t;

typedef xts_status_t (*xts_eval_fn)(void *user, const DLManagedTensorVersioned *x,
                                    double *value_out);
typedef xts_status_t (*xts_grad_fn)(void *user, const DLManagedTensorVersioned *x,
                                    DLManagedTensorVersioned *grad_out);

const char *xts_version(void);
const char *xts_last_error(void);
DLManagedTensorVersioned *xts_tensor_borrow_cpu_f64(double *data, size_t n);
void xts_tensor_free(DLManagedTensorVersioned *tensor);
xts_status_t xts_minimize(xts_eval_fn eval, xts_grad_fn grad, void *user,
                          DLManagedTensorVersioned *x, const xts_control_t *ctrl,
                          xts_method_t method, xts_report_t *out);

#ifdef __cplusplus
}
#endif

#endif /* XTS_OPTIMIZE_H */
