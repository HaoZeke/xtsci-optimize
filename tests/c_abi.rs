//! Drive the C waist from Rust (Rosenbrock 2D).

#![cfg(feature = "capi")]

use std::os::raw::c_void;

use quench_core::ffi::{
    quench_control_t, quench_method_t, quench_minimize_fn, quench_report_t,
    quench_status_t,
};

unsafe extern "C" fn rosen_eval(x: *const f64, n: usize, _user: *mut c_void) -> f64 {
    assert_eq!(n, 2);
    let x0 = unsafe { *x };
    let x1 = unsafe { *x.add(1) };
    100.0 * (x1 - x0 * x0).powi(2) + (1.0 - x0).powi(2)
}

unsafe extern "C" fn rosen_grad(x: *const f64, g: *mut f64, n: usize, _user: *mut c_void) {
    assert_eq!(n, 2);
    let x0 = unsafe { *x };
    let x1 = unsafe { *x.add(1) };
    let t = x1 - x0 * x0;
    unsafe {
        *g = -400.0 * x0 * t + 2.0 * (x0 - 1.0);
        *g.add(1) = 200.0 * t;
    }
}

#[test]
fn c_abi_lbfgs_rosenbrock() {
    let mut x = [-1.2, 1.0];
    let ctrl = quench_control_t {
        maxiter: 200,
        gtol: 1e-10,
        istep: 0.1,
        memory: 10,
    };
    let mut out = quench_report_t {
        value: 0.0,
        steps: 0,
        grad_norm: 0.0,
    };
    let st = unsafe {
        quench_minimize_fn(
            Some(rosen_eval as _),
            Some(rosen_grad as _),
            std::ptr::null_mut(),
            x.as_mut_ptr(),
            2,
            &ctrl,
            quench_method_t::QUENCH_LBFGS,
            &mut out,
        )
    };
    assert_eq!(st, quench_status_t::QUENCH_SUCCESS);
    assert!(out.value < 1e-6, "C ABI L-BFGS value {}", out.value);
    assert!((x[0] - 1.0).abs() < 1e-3);
    assert!((x[1] - 1.0).abs() < 1e-3);
}

#[test]
fn version_is_nul_terminated() {
    let p = quench_core::ffi::quench_version();
    assert!(!p.is_null());
}
