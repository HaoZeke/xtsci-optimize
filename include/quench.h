#ifndef QUENCH_H
#define QUENCH_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>
#include <dlpack/dlpack.h>

typedef enum quench_status_t {
    QUENCH_SUCCESS = 0,
    QUENCH_INVALID_PARAMETER = 1,
    QUENCH_INTERNAL_ERROR = 2,
    QUENCH_UNSUPPORTED_DEVICE = 3
} quench_status_t;

typedef enum quench_method_t {
    QUENCH_POLAK_RIBIERE = 0,
    QUENCH_FLETCHER_REEVES = 1,
    QUENCH_BFGS = 2,
    QUENCH_LBFGS = 3,
    QUENCH_SR1 = 4,
    QUENCH_ADAM = 5,
    QUENCH_STEEPEST = 6,
    QUENCH_SR2 = 7,
    QUENCH_PSO = 8,
    QUENCH_HESTENES_STIEFEL = 9,
    QUENCH_DAI_YUAN = 10,
    QUENCH_CONJUGATE_DESCENT = 11,
    QUENCH_HAGER_ZHANG = 12,
    QUENCH_LIU_STOREY = 13,
    QUENCH_FR_PR = 14
} quench_method_t;

typedef struct quench_control_t {
    size_t maxiter;
    double gtol;
    double istep;
    size_t memory;
} quench_control_t;

typedef struct quench_report_t {
    double value;
    size_t steps;
    double grad_norm;
} quench_report_t;

typedef quench_status_t (*quench_eval_fn)(void *user,
                                          const DLManagedTensorVersioned *x,
                                          double *value_out);
typedef quench_status_t (*quench_grad_fn)(void *user,
                                          const DLManagedTensorVersioned *x,
                                          DLManagedTensorVersioned *grad_out);

const char *quench_version(void);
const char *quench_last_error(void);
DLManagedTensorVersioned *quench_tensor_borrow_cpu_f64(double *data, size_t n);
void quench_tensor_free(DLManagedTensorVersioned *tensor);
quench_status_t quench_minimize_fn(quench_eval_fn eval, quench_grad_fn grad,
                                   void *user, DLManagedTensorVersioned *x,
                                   const quench_control_t *ctrl,
                                   quench_method_t method,
                                   quench_report_t *out);

#ifdef __cplusplus
}
#endif

#endif /* QUENCH_H */
