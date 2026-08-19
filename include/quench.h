#ifndef QUENCH_H
#define QUENCH_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>

typedef enum quench_status_t {
    QUENCH_SUCCESS = 0,
    QUENCH_INVALID_PARAMETER = 1,
    QUENCH_INTERNAL_ERROR = 2
} quench_status_t;

typedef enum quench_method_t {
    QUENCH_POLAK_RIBIERE = 0,
    QUENCH_FLETCHER_REEVES = 1,
    QUENCH_BFGS = 2,
    QUENCH_LBFGS = 3,
    QUENCH_SR1 = 4,
    QUENCH_ADAM = 5,
    QUENCH_STEEPEST = 6
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

typedef double (*quench_eval_fn)(const double *x, size_t n, void *user);
typedef void (*quench_grad_fn)(const double *x, double *g, size_t n, void *user);

const char *quench_version(void);
const char *quench_last_error(void);
quench_status_t quench_minimize_fn(quench_eval_fn eval, quench_grad_fn grad,
                                   void *user, double *x, size_t n,
                                   const quench_control_t *ctrl,
                                   quench_method_t method,
                                   quench_report_t *out);

#ifdef __cplusplus
}
#endif

#endif /* QUENCH_H */
