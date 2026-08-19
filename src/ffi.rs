//! Narrow C waist for quench. Algorithms stay in Rust; C, C++, and
//! xtensor talk only to this file.

use std::cell::RefCell;
use std::ffi::CString;
use std::os::raw::{c_char, c_void};
use std::slice;

use ndarray::Array1;

use crate::{Control, LineSearch, Method, Oracle, minimize_method};

/// Status codes. 0 is success, matching metatensor / eindir.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum quench_status_t {
    /// Completed.
    QUENCH_SUCCESS = 0,
    /// Null pointer or inconsistent length.
    QUENCH_INVALID_PARAMETER = 1,
    /// Panic or internal failure behind the C boundary.
    QUENCH_INTERNAL_ERROR = 2,
}

/// Method tag. Keep this a closed C enum; Rust [`Method`] is the source.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum quench_method_t {
    /// Polak-Ribiere NLCG + Brent.
    QUENCH_POLAK_RIBIERE = 0,
    /// Fletcher-Reeves NLCG + Brent.
    QUENCH_FLETCHER_REEVES = 1,
    /// Dense inverse-BFGS.
    QUENCH_BFGS = 2,
    /// Limited-memory BFGS.
    QUENCH_LBFGS = 3,
    /// Inverse SR1.
    QUENCH_SR1 = 4,
    /// Adam + Brent.
    QUENCH_ADAM = 5,
    /// Steepest descent.
    QUENCH_STEEPEST = 6,
}

/// Iteration controls. `memory` is used only by L-BFGS (0 means 10).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct quench_control_t {
    /// Maximum iterations.
    pub maxiter: usize,
    /// Stop when `||g||_2 < gtol`.
    pub gtol: f64,
    /// Initial line-search step.
    pub istep: f64,
    /// L-BFGS correction pairs; 0 selects 10.
    pub memory: usize,
}

/// Result written by [`quench_minimize_fn`].
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct quench_report_t {
    /// `f(x)` at the accepted point.
    pub value: f64,
    /// Outer iterations.
    pub steps: usize,
    /// `||∇f||_2`.
    pub grad_norm: f64,
}

/// `f(x)` callback. `x` has length `n`.
pub type quench_eval_fn =
    unsafe extern "C" fn(x: *const f64, n: usize, user: *mut c_void) -> f64;

/// `∇f(x)` callback. Writes `n` entries into `g`.
pub type quench_grad_fn =
    unsafe extern "C" fn(x: *const f64, g: *mut f64, n: usize, user: *mut c_void);

thread_local! {
    static LAST_ERROR: RefCell<CString> = RefCell::new(CString::default());
}

fn set_last_error(msg: &str) {
    LAST_ERROR.with(|cell| {
        let c = CString::new(msg)
            .unwrap_or_else(|_| CString::new("(interior NUL)").unwrap());
        *cell.borrow_mut() = c;
    });
}

/// Last error on this thread. Valid until the next quench C call.
#[unsafe(no_mangle)]
pub extern "C" fn quench_last_error() -> *const c_char {
    LAST_ERROR.with(|cell| cell.borrow().as_ptr())
}

/// Package version, NUL-terminated.
#[unsafe(no_mangle)]
pub extern "C" fn quench_version() -> *const c_char {
    concat!(env!("CARGO_PKG_VERSION"), "\0").as_ptr() as *const c_char
}

fn method_from_c(m: quench_method_t, memory: usize) -> Method {
    match m {
        quench_method_t::QUENCH_POLAK_RIBIERE => Method::polak_ribiere(),
        quench_method_t::QUENCH_FLETCHER_REEVES => Method::Nlcg {
            conjugacy: crate::Conjugacy::FletcherReeves,
            restart: crate::Restart::Never,
        },
        quench_method_t::QUENCH_BFGS => Method::Bfgs,
        quench_method_t::QUENCH_LBFGS => Method::Lbfgs {
            memory: if memory == 0 { 10 } else { memory },
        },
        quench_method_t::QUENCH_SR1 => Method::Sr1,
        quench_method_t::QUENCH_ADAM => Method::adam(),
        quench_method_t::QUENCH_STEEPEST => Method::Steepest,
    }
}

/// Minimize `f` from `x[0..n]`. On success, `x` holds the accepted point
/// and `out` is filled.
///
/// # Safety
///
/// `eval` and `grad` must be callable for the lifetime of this call. `x`
/// must point to `n` writable doubles. `ctrl` and `out` must be non-null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn quench_minimize_fn(
    eval: Option<quench_eval_fn>,
    grad: Option<quench_grad_fn>,
    user: *mut c_void,
    x: *mut f64,
    n: usize,
    ctrl: *const quench_control_t,
    method: quench_method_t,
    out: *mut quench_report_t,
) -> quench_status_t {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let eval = match eval {
            Some(f) => f,
            None => {
                set_last_error("quench_minimize_fn: eval is NULL");
                return quench_status_t::QUENCH_INVALID_PARAMETER;
            }
        };
        let grad = match grad {
            Some(f) => f,
            None => {
                set_last_error("quench_minimize_fn: grad is NULL");
                return quench_status_t::QUENCH_INVALID_PARAMETER;
            }
        };
        if x.is_null() || n == 0 || ctrl.is_null() || out.is_null() {
            set_last_error("quench_minimize_fn: x/ctrl/out null or n==0");
            return quench_status_t::QUENCH_INVALID_PARAMETER;
        }
        let c = unsafe { &*ctrl };
        let init = unsafe { slice::from_raw_parts(x, n) }.to_vec();
        let eval_ptr = eval as usize;
        let grad_ptr = grad as usize;
        let user_addr = user as usize;
        let obj = Oracle::unbounded(n, move |xv| {
            let eval_fn: quench_eval_fn = unsafe { std::mem::transmute(eval_ptr) };
            let grad_fn: quench_grad_fn = unsafe { std::mem::transmute(grad_ptr) };
            let user = user_addr as *mut c_void;
            let xs = xv.as_slice().expect("contiguous");
            let value = unsafe { eval_fn(xs.as_ptr(), xs.len(), user) };
            let mut g = vec![0.0; xs.len()];
            unsafe { grad_fn(xs.as_ptr(), g.as_mut_ptr(), xs.len(), user) };
            (value, Array1::from(g))
        });
        let control = Control {
            maxiter: c.maxiter,
            gtol: c.gtol,
            istep: if c.istep > 0.0 { c.istep } else { 1.0 },
            maxmove: None,
        };
        match minimize_method(
            &obj,
            Array1::from(init),
            &control,
            method_from_c(method, c.memory),
            LineSearch::default(),
        ) {
            Ok(rep) => {
                let dest = unsafe { slice::from_raw_parts_mut(x, n) };
                dest.copy_from_slice(rep.coords.as_slice().expect("contiguous"));
                unsafe {
                    *out = quench_report_t {
                        value: rep.value,
                        steps: rep.steps,
                        grad_norm: rep.grad_norm,
                    };
                }
                quench_status_t::QUENCH_SUCCESS
            }
            Err(e) => {
                set_last_error(&e.to_string());
                quench_status_t::QUENCH_INVALID_PARAMETER
            }
        }
    })) {
        Ok(s) => s,
        Err(_) => {
            set_last_error("quench_minimize_fn: panic");
            quench_status_t::QUENCH_INTERNAL_ERROR
        }
    }
}
