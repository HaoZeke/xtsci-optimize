//! Drive the DLPack C waist from Rust (Rosenbrock 2D).

#![cfg(feature = "capi")]

use std::os::raw::c_void;

use dlpk::sys::{DLDeviceType, DLManagedTensorVersioned};
use eindir_core::ffi::eindir_core_abi_stamp;
use eindir_core::ffi::{eindir_objective_t, eindir_status_t};
use rgpot_core::eindir::{rgpot_potential_free_eindir, rgpot_potential_new_eindir};
use rgpot_core::status::rgpot_status_t;
use rgpot_core::types::{rgpot_force_input_t, rgpot_force_out_t};
use xtsci_optimize::ffi::{
    xts_abi_compatible, xts_abi_stamp, xts_accept_t, xts_control_t, xts_method_t, xts_minimize,
    xts_minimize_eindir, xts_report_t, xts_solver_create, xts_solver_free, xts_solver_set_accept,
    xts_solver_step, xts_solver_step_fg, xts_status_t, xts_tensor_borrow_cpu_f64, xts_tensor_free,
};

unsafe extern "C" fn quadratic_eval(
    _user: *mut c_void,
    x: *const DLManagedTensorVersioned,
    value_out: *mut f64,
) -> eindir_status_t {
    let (p, n) = unsafe { cpu_f64(x) };
    assert_eq!(n, 2);
    unsafe { *value_out = *p * *p + *p.add(1) * *p.add(1) };
    eindir_status_t::EINDIR_SUCCESS
}

unsafe extern "C" fn quadratic_grad(
    _user: *mut c_void,
    x: *const DLManagedTensorVersioned,
    g: *mut DLManagedTensorVersioned,
) -> eindir_status_t {
    let (p, n) = unsafe { cpu_f64(x) };
    assert_eq!(n, 2);
    let (gp, gn) = unsafe { cpu_f64(g) };
    assert_eq!(gn, 2);
    unsafe {
        *(gp as *mut f64) = 2.0 * *p;
        *(gp as *mut f64).add(1) = 2.0 * *p.add(1);
    }
    eindir_status_t::EINDIR_SUCCESS
}

fn quadratic_objective() -> Box<eindir_objective_t> {
    Box::new(eindir_objective_t {
        dim: 2,
        low: std::ptr::null_mut(),
        high: std::ptr::null_mut(),
        eval_fn: quadratic_eval,
        grad_fn: Some(quadratic_grad),
        user_data: std::ptr::null_mut(),
        free_fn: None,
    })
}

unsafe fn cpu_f64(t: *const DLManagedTensorVersioned) -> (*const f64, usize) {
    let dl = unsafe { &(*t).dl_tensor };
    let n = unsafe { *dl.shape as usize };
    let p = unsafe { (dl.data as *const u8).add(dl.byte_offset as usize) as *const f64 };
    (p, n)
}

unsafe extern "C" fn rosen_eval(
    _user: *mut c_void,
    x: *const DLManagedTensorVersioned,
    value_out: *mut f64,
) -> xts_status_t {
    let (p, n) = unsafe { cpu_f64(x) };
    assert_eq!(n, 2);
    let x0 = unsafe { *p };
    let x1 = unsafe { *p.add(1) };
    unsafe {
        *value_out = 100.0 * (x1 - x0 * x0).powi(2) + (1.0 - x0).powi(2);
    }
    xts_status_t::XTS_SUCCESS
}

unsafe extern "C" fn rosen_evalgrad(
    user: *mut c_void,
    x: *const DLManagedTensorVersioned,
    value_out: *mut f64,
    g: *mut DLManagedTensorVersioned,
) -> xts_status_t {
    let n = unsafe { &mut *(user as *mut usize) };
    *n += 1;
    let ev = rosen_eval(std::ptr::null_mut(), x, value_out);
    if ev != xts_status_t::XTS_SUCCESS {
        return ev;
    }
    rosen_grad(std::ptr::null_mut(), x, g)
}

unsafe extern "C" fn rosen_grad(
    _user: *mut c_void,
    x: *const DLManagedTensorVersioned,
    g: *mut DLManagedTensorVersioned,
) -> xts_status_t {
    let (p, n) = unsafe { cpu_f64(x) };
    assert_eq!(n, 2);
    let x0 = unsafe { *p };
    let x1 = unsafe { *p.add(1) };
    let t = x1 - x0 * x0;
    let (gp, gn) = unsafe { cpu_f64(g as *const _) };
    assert_eq!(gn, 2);
    unsafe {
        *(gp as *mut f64) = -400.0 * x0 * t + 2.0 * (x0 - 1.0);
        *(gp as *mut f64).add(1) = 200.0 * t;
    }
    xts_status_t::XTS_SUCCESS
}

#[test]
fn c_abi_lbfgs_rosenbrock() {
    let mut x = [-1.2, 1.0];
    let xt = unsafe { xts_tensor_borrow_cpu_f64(x.as_mut_ptr(), 2) };
    assert!(!xt.is_null());
    let ctrl = xts_control_t {
        maxiter: 200,
        gtol: 1e-10,
        istep: 0.1,
        memory: 10,
        maxmove: 0.0,
    };
    let mut out = xts_report_t {
        value: 0.0,
        steps: 0,
        grad_norm: 0.0,
    };
    let st = unsafe {
        xts_minimize(
            Some(rosen_eval),
            Some(rosen_grad),
            std::ptr::null_mut(),
            xt,
            &ctrl,
            xts_method_t::XTS_LBFGS,
            &mut out,
        )
    };
    unsafe { xts_tensor_free(xt) };
    assert_eq!(st, xts_status_t::XTS_SUCCESS);
    assert!(out.value < 1e-6, "C ABI L-BFGS value {}", out.value);
    assert!((x[0] - 1.0).abs() < 1e-3);
    assert!((x[1] - 1.0).abs() < 1e-3);
}

#[test]
fn c_abi_eindir_objective_minimizes_without_taking_ownership() {
    let mut x = [3.0, -4.0];
    let xt = unsafe { xts_tensor_borrow_cpu_f64(x.as_mut_ptr(), 2) };
    assert!(!xt.is_null());
    let objective = quadratic_objective();
    let ctrl = xts_control_t {
        maxiter: 100,
        gtol: 1e-10,
        istep: 0.1,
        memory: 10,
        maxmove: 0.0,
    };
    let mut out = xts_report_t {
        value: 0.0,
        steps: 0,
        grad_norm: 0.0,
    };
    let st = unsafe {
        xts_minimize_eindir(
            &*objective,
            &eindir_core_abi_stamp(),
            xt,
            &ctrl,
            xts_method_t::XTS_LBFGS,
            &mut out,
        )
    };
    assert_eq!(st, xts_status_t::XTS_SUCCESS);
    assert!(out.value < 1e-12, "eindir objective value {}", out.value);
    assert!(x.iter().all(|value| value.abs() < 1e-5));
    assert_eq!(objective.dim, 2);
    unsafe { xts_tensor_free(xt) };
}

#[test]
fn c_abi_eindir_rejects_incompatible_stamp_before_evaluation() {
    let mut x = [3.0, -4.0];
    let xt = unsafe { xts_tensor_borrow_cpu_f64(x.as_mut_ptr(), 2) };
    let objective = quadratic_objective();
    let mut stamp = eindir_core_abi_stamp();
    stamp.objective_layout += 1;
    let ctrl = xts_control_t {
        maxiter: 1,
        gtol: 1e-10,
        istep: 0.1,
        memory: 10,
        maxmove: 0.0,
    };
    let mut out = xts_report_t {
        value: 0.0,
        steps: 0,
        grad_norm: 0.0,
    };
    let st = unsafe {
        xts_minimize_eindir(
            &*objective,
            &stamp,
            xt,
            &ctrl,
            xts_method_t::XTS_LBFGS,
            &mut out,
        )
    };
    assert_eq!(st, xts_status_t::XTS_INVALID_PARAMETER);
    assert_eq!(x, [3.0, -4.0]);
    unsafe { xts_tensor_free(xt) };
}

struct QuadraticContext {
    forces: [f64; 6],
}

unsafe extern "C" fn rgpot_quadratic_callback(
    user: *mut c_void,
    input: *const rgpot_force_input_t,
    output: *mut rgpot_force_out_t,
) -> rgpot_status_t {
    let context = unsafe { &mut *(user as *mut QuadraticContext) };
    let input = unsafe { &*input };
    let output = unsafe { &mut *output };
    let positions = unsafe { &(*input.positions).dl_tensor };
    let n = unsafe { (positions.shape as *const i64).read() as usize * 3 };
    let values = unsafe { std::slice::from_raw_parts(positions.data as *const f64, n) };
    output.energy = values.iter().map(|value| value * value).sum();
    context
        .forces
        .iter_mut()
        .zip(values)
        .for_each(|(force, value)| *force = -2.0 * value);
    output.forces = unsafe {
        rgpot_core::tensor::rgpot_tensor_cpu_f64_2d(context.forces.as_mut_ptr(), n as i64 / 3, 3)
    };
    rgpot_status_t::RGPOT_SUCCESS
}

#[test]
fn c_abi_rgpot_potential_reaches_eindir_optimizer() {
    let atomic_numbers = [1i32, 1];
    let cell = [10.0f64, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0];
    let mut context = QuadraticContext { forces: [0.0; 6] };
    let potential = unsafe {
        rgpot_potential_new_eindir(
            rgpot_quadratic_callback,
            (&mut context as *mut QuadraticContext).cast(),
            None,
            2,
            atomic_numbers.as_ptr(),
            cell.as_ptr(),
            std::ptr::null(),
            std::ptr::null(),
        )
    };
    assert!(!potential.is_null());

    let mut x = [2.0, -1.5, 0.5, -2.5, 1.0, -0.25];
    let xt = unsafe { xts_tensor_borrow_cpu_f64(x.as_mut_ptr(), x.len()) };
    let ctrl = xts_control_t {
        maxiter: 100,
        gtol: 1e-8,
        istep: 0.1,
        memory: 10,
        maxmove: 0.0,
    };
    let mut out = xts_report_t {
        value: 0.0,
        steps: 0,
        grad_norm: 0.0,
    };
    let stamp = rgpot_core::eindir::rgpot_eindir_abi_stamp();
    let status = unsafe {
        xts_minimize_eindir(
            potential.cast(),
            &stamp,
            xt,
            &ctrl,
            xts_method_t::XTS_LBFGS,
            &mut out,
        )
    };

    assert_eq!(status, xts_status_t::XTS_SUCCESS);
    assert!(out.value < 1e-12, "rgpot objective value {}", out.value);
    assert!(x.iter().all(|value| value.abs() < 1e-5));
    unsafe {
        xts_tensor_free(xt);
        rgpot_potential_free_eindir(potential);
    }
}

#[test]
fn cuda_tagged_tensor_is_unsupported() {
    let mut x = [-1.2, 1.0];
    let xt = unsafe { xts_tensor_borrow_cpu_f64(x.as_mut_ptr(), 2) };
    assert!(!xt.is_null());
    unsafe {
        (*xt).dl_tensor.device.device_type = DLDeviceType::kDLCUDA;
    }
    let ctrl = xts_control_t {
        maxiter: 1,
        gtol: 1e-10,
        istep: 0.1,
        memory: 10,
        maxmove: 0.0,
    };
    let mut out = xts_report_t {
        value: 0.0,
        steps: 0,
        grad_norm: 0.0,
    };
    let st = unsafe {
        xts_minimize(
            Some(rosen_eval),
            Some(rosen_grad),
            std::ptr::null_mut(),
            xt,
            &ctrl,
            xts_method_t::XTS_LBFGS,
            &mut out,
        )
    };
    unsafe { xts_tensor_free(xt) };
    assert_eq!(st, xts_status_t::XTS_UNSUPPORTED_DEVICE);
}

#[test]
fn c_abi_solver_step_keeps_lbfgs_history() {
    let mut warm = [-1.2, 1.0];
    let mut cold = [-1.2, 1.0];
    let ctrl = xts_control_t {
        maxiter: 80,
        gtol: 1e-10,
        istep: 0.1,
        memory: 10,
        maxmove: 0.0,
    };
    let session = unsafe { xts_solver_create(xts_method_t::XTS_LBFGS, &ctrl, 2) };
    assert!(!session.is_null());
    let mut out = xts_report_t {
        value: 0.0,
        steps: 0,
        grad_norm: 0.0,
    };
    for _ in 0..80 {
        let xt = unsafe { xts_tensor_borrow_cpu_f64(warm.as_mut_ptr(), 2) };
        let st = unsafe {
            xts_solver_step(
                session,
                Some(rosen_eval),
                Some(rosen_grad),
                std::ptr::null_mut(),
                xt,
                &mut out,
            )
        };
        unsafe { xts_tensor_free(xt) };
        assert_eq!(st, xts_status_t::XTS_SUCCESS);
        if out.grad_norm < 1e-8 {
            break;
        }
    }
    unsafe { xts_solver_free(session) };
    assert!(out.value < 1e-6, "session value {}", out.value);

    for _ in 0..80 {
        let one = unsafe { xts_solver_create(xts_method_t::XTS_LBFGS, &ctrl, 2) };
        let xt = unsafe { xts_tensor_borrow_cpu_f64(cold.as_mut_ptr(), 2) };
        let st = unsafe {
            xts_solver_step(
                one,
                Some(rosen_eval),
                Some(rosen_grad),
                std::ptr::null_mut(),
                xt,
                &mut out,
            )
        };
        unsafe {
            xts_tensor_free(xt);
            xts_solver_free(one);
        }
        assert_eq!(st, xts_status_t::XTS_SUCCESS);
        if out.grad_norm < 1e-8 {
            break;
        }
    }
    // A live session is L-BFGS. Recreating every step is steepest-plus-Wolfe.
    assert!(
        (warm[0] - 1.0).abs() < 1e-3,
        "warm end {} {}",
        warm[0],
        warm[1]
    );
}

#[test]
fn version_is_nul_terminated() {
    let p = xtsci_optimize::ffi::xts_version();
    assert!(!p.is_null());
}

#[test]
fn fused_evalgrad_is_one_callback_per_oracle() {
    let mut x = [-1.2, 1.0];
    let xt = unsafe { xts_tensor_borrow_cpu_f64(x.as_mut_ptr(), 2) };
    let ctrl = xts_control_t {
        maxiter: 1,
        gtol: 1e-12,
        istep: 0.1,
        memory: 10,
        maxmove: 0.2,
    };
    let session = unsafe { xts_solver_create(xts_method_t::XTS_LBFGS, &ctrl, 2) };
    assert!(!session.is_null());
    unsafe { xts_solver_set_accept(session, xts_accept_t::XTS_ACCEPT_NONE) };
    let mut calls = 0usize;
    let mut out = xts_report_t {
        value: 0.0,
        steps: 0,
        grad_norm: 0.0,
    };
    let st = unsafe {
        xts_solver_step_fg(
            session,
            Some(rosen_evalgrad),
            (&mut calls as *mut usize).cast(),
            xt,
            &mut out,
        )
    };
    unsafe {
        xts_solver_free(session);
        xts_tensor_free(xt);
    }
    assert_eq!(st, xts_status_t::XTS_SUCCESS);
    assert!(calls >= 1, "fused oracle never ran");
}

#[test]
fn abi_stamp_identifies_this_optimizer_layout() {
    let stamp = xts_abi_stamp();
    assert_eq!(stamp.abi_major, 1);
    assert_eq!(stamp.abi_minor, 10);
    assert_eq!(stamp.layout_revision, 2);
    assert_eq!(unsafe { xts_abi_compatible(&stamp) }, 1);
}

#[test]
fn abi_stamp_rejects_an_incompatible_layout() {
    let mut stamp = xts_abi_stamp();
    stamp.layout_revision += 1;
    assert_eq!(unsafe { xts_abi_compatible(&stamp) }, 0);
}

#[test]
fn c_abi_respects_maxmove_when_initial_step_is_larger() {
    let mut x = [1.0, 0.0];
    let xt = unsafe { xts_tensor_borrow_cpu_f64(x.as_mut_ptr(), 2) };
    let ctrl = xts_control_t {
        maxiter: 1,
        gtol: 1e-12,
        istep: 10.0,
        memory: 10,
        maxmove: 0.1,
    };
    let mut out = xts_report_t {
        value: 0.0,
        steps: 0,
        grad_norm: 0.0,
    };
    let st = unsafe {
        xts_minimize(
            Some(rosen_eval),
            Some(rosen_grad),
            std::ptr::null_mut(),
            xt,
            &ctrl,
            xts_method_t::XTS_LBFGS,
            &mut out,
        )
    };
    unsafe { xts_tensor_free(xt) };
    assert_eq!(st, xts_status_t::XTS_SUCCESS);
    assert!((1.0 - x[0]).abs() <= 0.1 + 1e-12);
}
