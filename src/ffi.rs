//! Narrow C waist for quench. Every vector is a DLPack tensor (dlpk),
//! the same contract as eindir_objective_t, so a later device kernel
//! does not change the ABI.

use std::cell::RefCell;
use std::ffi::CString;
use std::os::raw::{c_char, c_void};
use std::slice;

use dlpk::sys::{
    DLDataType, DLDataTypeCode, DLDevice, DLDeviceType, DLManagedTensorVersioned, DLPackVersion,
    DLTensor,
};
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
    /// Tensor is not on a device this build can evaluate (GPU later).
    QUENCH_UNSUPPORTED_DEVICE = 3,
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
    /// SR2 Hessian update.
    QUENCH_SR2 = 7,
    /// Particle swarm.
    QUENCH_PSO = 8,
    /// Hestenes-Stiefel NLCG + Brent.
    QUENCH_HESTENES_STIEFEL = 9,
    /// Dai-Yuan NLCG + Brent.
    QUENCH_DAI_YUAN = 10,
    /// Fletcher conjugate-descent NLCG + Brent.
    QUENCH_CONJUGATE_DESCENT = 11,
    /// Hager-Zhang NLCG + Brent.
    QUENCH_HAGER_ZHANG = 12,
    /// Liu-Storey NLCG + Brent.
    QUENCH_LIU_STOREY = 13,
    /// Gilbert-Nocedal FR-PR hybrid NLCG + Brent.
    QUENCH_FR_PR = 14,
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

/// `f(x)` callback. `x` is a rank-1 f64 DLPack tensor.
pub type quench_eval_fn = unsafe extern "C" fn(
    user: *mut c_void,
    x: *const DLManagedTensorVersioned,
    value_out: *mut f64,
) -> quench_status_t;

/// `∇f(x)` callback. Writes into the pre-allocated `grad_out` tensor.
pub type quench_grad_fn = unsafe extern "C" fn(
    user: *mut c_void,
    x: *const DLManagedTensorVersioned,
    grad_out: *mut DLManagedTensorVersioned,
) -> quench_status_t;

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
        quench_method_t::QUENCH_FLETCHER_REEVES => {
            Method::nlcg(crate::Conjugacy::FletcherReeves)
        }
        quench_method_t::QUENCH_BFGS => Method::Bfgs,
        quench_method_t::QUENCH_LBFGS => Method::Lbfgs {
            memory: if memory == 0 { 10 } else { memory },
        },
        quench_method_t::QUENCH_SR1 => Method::Sr1,
        quench_method_t::QUENCH_ADAM => Method::adam(),
        quench_method_t::QUENCH_STEEPEST => Method::Steepest,
        quench_method_t::QUENCH_SR2 => Method::Sr2,
        quench_method_t::QUENCH_PSO => Method::pso(),
        quench_method_t::QUENCH_HESTENES_STIEFEL => {
            Method::nlcg(crate::Conjugacy::HestenesStiefel)
        }
        quench_method_t::QUENCH_DAI_YUAN => Method::nlcg(crate::Conjugacy::DaiYuan),
        quench_method_t::QUENCH_CONJUGATE_DESCENT => {
            Method::nlcg(crate::Conjugacy::ConjugateDescent)
        }
        quench_method_t::QUENCH_HAGER_ZHANG => Method::nlcg(crate::Conjugacy::HagerZhang),
        quench_method_t::QUENCH_LIU_STOREY => Method::nlcg(crate::Conjugacy::LiuStorey),
        quench_method_t::QUENCH_FR_PR => Method::nlcg(crate::Conjugacy::FrPr),
    }
}

/// Borrow a 1-D CPU f64 buffer as a DLPack tensor. Caller must
/// [`quench_tensor_free`] it. The buffer must outlive the tensor.
///
/// # Safety
/// `data` points to `len` writable f64s.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn quench_tensor_borrow_cpu_f64(
    data: *mut f64,
    len: usize,
) -> *mut DLManagedTensorVersioned {
    if data.is_null() || len == 0 {
        set_last_error("quench_tensor_borrow_cpu_f64: null or empty");
        return std::ptr::null_mut();
    }
    unsafe { create_borrowed_f64_1d(data, len, DLDeviceType::kDLCPU, 0) }
}

/// Release a tensor created by [`quench_tensor_borrow_cpu_f64`].
///
/// # Safety
/// `tensor` is null or a pointer from this crate's borrow helper.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn quench_tensor_free(tensor: *mut DLManagedTensorVersioned) {
    if tensor.is_null() {
        return;
    }
    if let Some(deleter) = unsafe { (*tensor).deleter } {
        unsafe { deleter(tensor) };
    }
}

unsafe fn create_borrowed_f64_1d(
    data: *mut f64,
    len: usize,
    device_type: DLDeviceType,
    device_id: i32,
) -> *mut DLManagedTensorVersioned {
    struct Ctx {
        shape: [i64; 1],
        strides: [i64; 1],
    }

    unsafe extern "C" fn deleter(ptr: *mut DLManagedTensorVersioned) {
        if ptr.is_null() {
            return;
        }
        let ctx = unsafe { (*ptr).manager_ctx.cast::<Ctx>() };
        if !ctx.is_null() {
            drop(unsafe { Box::from_raw(ctx) });
        }
        drop(unsafe { Box::from_raw(ptr) });
    }

    let mut ctx = Box::new(Ctx {
        shape: [len as i64],
        strides: [1],
    });
    let dl_tensor = DLTensor {
        data: data.cast(),
        device: DLDevice {
            device_type,
            device_id,
        },
        ndim: 1,
        dtype: DLDataType {
            code: DLDataTypeCode::kDLFloat,
            bits: 64,
            lanes: 1,
        },
        shape: ctx.shape.as_mut_ptr(),
        strides: ctx.strides.as_mut_ptr(),
        byte_offset: 0,
    };
    let managed = Box::new(DLManagedTensorVersioned {
        version: DLPackVersion { major: 1, minor: 0 },
        manager_ctx: Box::into_raw(ctx).cast(),
        deleter: Some(deleter),
        flags: 0,
        dl_tensor,
    });
    Box::into_raw(managed)
}

fn cpu_f64_slice<'a>(
    t: *const DLManagedTensorVersioned,
    name: &str,
) -> Result<&'a [f64], quench_status_t> {
    if t.is_null() {
        set_last_error(&format!("{name}: null tensor"));
        return Err(quench_status_t::QUENCH_INVALID_PARAMETER);
    }
    let t = unsafe { &*t };
    let dl = &t.dl_tensor;
    if dl.device.device_type != DLDeviceType::kDLCPU {
        set_last_error(&format!(
            "{name}: device {:?} not supported in this build (CPU only)",
            dl.device.device_type as i32
        ));
        return Err(quench_status_t::QUENCH_UNSUPPORTED_DEVICE);
    }
    if dl.ndim != 1
        || dl.dtype.code != DLDataTypeCode::kDLFloat
        || dl.dtype.bits != 64
        || dl.dtype.lanes != 1
        || dl.shape.is_null()
        || dl.data.is_null()
    {
        set_last_error(&format!("{name}: need rank-1 f64 contiguous DLPack"));
        return Err(quench_status_t::QUENCH_INVALID_PARAMETER);
    }
    let n = unsafe { *dl.shape as usize };
    if n == 0 {
        set_last_error(&format!("{name}: empty"));
        return Err(quench_status_t::QUENCH_INVALID_PARAMETER);
    }
    if !dl.strides.is_null() && unsafe { *dl.strides } != 1 {
        set_last_error(&format!("{name}: non-unit stride"));
        return Err(quench_status_t::QUENCH_INVALID_PARAMETER);
    }
    let ptr = unsafe { (dl.data as *const u8).add(dl.byte_offset as usize) as *const f64 };
    Ok(unsafe { slice::from_raw_parts(ptr, n) })
}

fn cpu_f64_slice_mut<'a>(
    t: *mut DLManagedTensorVersioned,
    name: &str,
) -> Result<&'a mut [f64], quench_status_t> {
    let s = cpu_f64_slice(t as *const _, name)?;
    let n = s.len();
    let p = s.as_ptr() as *mut f64;
    Ok(unsafe { slice::from_raw_parts_mut(p, n) })
}

/// Minimize `f` from the 1-D f64 DLPack iterate `x`. On success, `x` is
/// overwritten with the accepted point.
///
/// # Safety
///
/// `eval` and `grad` must be callable for the lifetime of this call.
/// `x` must be a writable rank-1 f64 tensor. Non-CPU devices return
/// [`quench_status_t::QUENCH_UNSUPPORTED_DEVICE`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn quench_minimize_fn(
    eval: Option<quench_eval_fn>,
    grad: Option<quench_grad_fn>,
    user: *mut c_void,
    x: *mut DLManagedTensorVersioned,
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
        if ctrl.is_null() || out.is_null() {
            set_last_error("quench_minimize_fn: ctrl/out null");
            return quench_status_t::QUENCH_INVALID_PARAMETER;
        }
        let init = match cpu_f64_slice_mut(x, "x") {
            Ok(s) => s.to_vec(),
            Err(st) => return st,
        };
        let n = init.len();
        let eval_ptr = eval as usize;
        let grad_ptr = grad as usize;
        let user_addr = user as usize;
        let obj = Oracle::unbounded(n, move |xv| {
            let eval_fn: quench_eval_fn = unsafe { std::mem::transmute(eval_ptr) };
            let grad_fn: quench_grad_fn = unsafe { std::mem::transmute(grad_ptr) };
            let user = user_addr as *mut c_void;
            let mut xs = xv.to_vec();
            let xt = unsafe {
                create_borrowed_f64_1d(xs.as_mut_ptr(), xs.len(), DLDeviceType::kDLCPU, 0)
            };
            let mut value = 0.0;
            let ev_st = unsafe { eval_fn(user, xt, &mut value) };
            let mut g = vec![0.0; xs.len()];
            let gt = unsafe {
                create_borrowed_f64_1d(g.as_mut_ptr(), g.len(), DLDeviceType::kDLCPU, 0)
            };
            let gr_st = unsafe { grad_fn(user, xt, gt) };
            unsafe {
                quench_tensor_free(xt);
                quench_tensor_free(gt);
            }
            if ev_st != quench_status_t::QUENCH_SUCCESS
                || gr_st != quench_status_t::QUENCH_SUCCESS
            {
                return (f64::INFINITY, Array1::from(g));
            }
            (value, Array1::from(g))
        });
        let c = unsafe { &*ctrl };
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
                let dest = match cpu_f64_slice_mut(x, "x") {
                    Ok(s) => s,
                    Err(st) => return st,
                };
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

#[cfg(test)]
mod device_tests {
    use super::*;

    #[test]
    fn cuda_tag_is_unsupported() {
        let mut buf = [0.0_f64; 2];
        let t = unsafe {
            create_borrowed_f64_1d(buf.as_mut_ptr(), 2, DLDeviceType::kDLCUDA, 0)
        };
        let err = cpu_f64_slice(t, "cuda").unwrap_err();
        unsafe { quench_tensor_free(t) };
        assert_eq!(err, quench_status_t::QUENCH_UNSUPPORTED_DEVICE);
    }
}
