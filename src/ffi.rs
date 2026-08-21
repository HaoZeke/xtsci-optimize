//! Narrow C waist for xtsci-optimize. Every vector is a DLPack tensor (dlpk),
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
use ndarray::{Array1, Array2};
use eindir_core::ffi::{
    eindir_abi_stamp_t, eindir_core_abi_compatible, eindir_objective_eval,
    eindir_objective_grad, eindir_objective_has_grad, eindir_objective_t,
    eindir_status_t,
};

use crate::{
    Control, HessianOracle, LineSearch, Method, NewtonKind, Oracle, minimize_method,
    minimize_newton,
};

/// Status codes. 0 is success, matching metatensor / eindir.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum xts_status_t {
    /// Completed.
    XTS_SUCCESS = 0,
    /// Null pointer or inconsistent length.
    XTS_INVALID_PARAMETER = 1,
    /// Panic or internal failure behind the C boundary.
    XTS_INTERNAL_ERROR = 2,
    /// Tensor is not on a device this build can evaluate (GPU later).
    XTS_UNSUPPORTED_DEVICE = 3,
}

/// Compatibility identity for the xtsci-optimize C ABI.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct xts_abi_stamp_t {
    /// Incompatible changes increment this value.
    pub abi_major: u16,
    /// Additive compatible changes increment this value.
    pub abi_minor: u16,
    /// Struct and function-layout revision for this ABI major/minor.
    pub layout_revision: u16,
}

pub const XTS_ABI_VERSION_MAJOR: u16 = 1;
pub const XTS_ABI_VERSION_MINOR: u16 = 1;
pub const XTS_ABI_LAYOUT_REVISION: u16 = 2;

/// Method tag. Keep this a closed C enum; Rust [`Method`] is the source.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum xts_method_t {
    /// Polak-Ribiere NLCG + Brent.
    XTS_POLAK_RIBIERE = 0,
    /// Fletcher-Reeves NLCG + Brent.
    XTS_FLETCHER_REEVES = 1,
    /// Dense inverse-BFGS.
    XTS_BFGS = 2,
    /// Limited-memory BFGS.
    XTS_LBFGS = 3,
    /// Inverse SR1.
    XTS_SR1 = 4,
    /// Adam + Brent.
    XTS_ADAM = 5,
    /// Steepest descent.
    XTS_STEEPEST = 6,
    /// SR2 Hessian update.
    XTS_SR2 = 7,
    /// Particle swarm.
    XTS_PSO = 8,
    /// Hestenes-Stiefel NLCG + Brent.
    XTS_HESTENES_STIEFEL = 9,
    /// Dai-Yuan NLCG + Brent.
    XTS_DAI_YUAN = 10,
    /// Fletcher conjugate-descent NLCG + Brent.
    XTS_CONJUGATE_DESCENT = 11,
    /// Hager-Zhang NLCG + Brent.
    XTS_HAGER_ZHANG = 12,
    /// Liu-Storey NLCG + Brent.
    XTS_LIU_STOREY = 13,
    /// Gilbert-Nocedal FR-PR hybrid NLCG + Brent.
    XTS_FR_PR = 14,
    /// Shifted Newton on a caller-supplied Hessian.
    XTS_NEWTON = 15,
    /// Banerjee / Baker RFO on a caller-supplied Hessian.
    XTS_RFO = 16,
}

/// Iteration controls. `memory` is used only by L-BFGS (0 means 10).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct xts_control_t {
    /// Maximum iterations.
    pub maxiter: usize,
    /// Stop when `||g||_2 < gtol`.
    pub gtol: f64,
    /// Initial line-search step.
    pub istep: f64,
    /// L-BFGS correction pairs; 0 selects 10.
    pub memory: usize,
    /// Euclidean cap on each proposed step; non-positive disables the cap.
    pub maxmove: f64,
}

/// Result written by [`xts_minimize`].
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct xts_report_t {
    /// `f(x)` at the accepted point.
    pub value: f64,
    /// Outer iterations.
    pub steps: usize,
    /// `||∇f||_2`.
    pub grad_norm: f64,
}

/// `f(x)` callback. `x` is a rank-1 f64 DLPack tensor.
pub type xts_eval_fn = unsafe extern "C" fn(
    user: *mut c_void,
    x: *const DLManagedTensorVersioned,
    value_out: *mut f64,
) -> xts_status_t;

/// `∇f(x)` callback. Writes into the pre-allocated `grad_out` tensor.
pub type xts_grad_fn = unsafe extern "C" fn(
    user: *mut c_void,
    x: *const DLManagedTensorVersioned,
    grad_out: *mut DLManagedTensorVersioned,
) -> xts_status_t;

/// `∇²f(x)` callback. Writes a length-`n²` row-major Hessian into `hess_out`.
pub type xts_hess_fn = unsafe extern "C" fn(
    user: *mut c_void,
    x: *const DLManagedTensorVersioned,
    hess_out: *mut DLManagedTensorVersioned,
) -> xts_status_t;

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

/// Last error on this thread. Valid until the next xts C call.
#[unsafe(no_mangle)]
pub extern "C" fn xts_last_error() -> *const c_char {
    LAST_ERROR.with(|cell| cell.borrow().as_ptr())
}

/// Package version, NUL-terminated.
#[unsafe(no_mangle)]
pub extern "C" fn xts_version() -> *const c_char {
    concat!(env!("CARGO_PKG_VERSION"), "\0").as_ptr() as *const c_char
}

/// Return the compatibility identity for this C ABI build.
#[unsafe(no_mangle)]
pub extern "C" fn xts_abi_stamp() -> xts_abi_stamp_t {
    xts_abi_stamp_t {
        abi_major: XTS_ABI_VERSION_MAJOR,
        abi_minor: XTS_ABI_VERSION_MINOR,
        layout_revision: XTS_ABI_LAYOUT_REVISION,
    }
}

/// Return nonzero when a caller's ABI identity is accepted by this build.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn xts_abi_compatible(stamp: *const xts_abi_stamp_t) -> i32 {
    if stamp.is_null() {
        return 0;
    }
    let stamp = unsafe { &*stamp };
    i32::from(
        stamp.abi_major == XTS_ABI_VERSION_MAJOR
            && stamp.layout_revision == XTS_ABI_LAYOUT_REVISION,
    )
}

fn method_from_c(m: xts_method_t, memory: usize) -> Method {
    match m {
        xts_method_t::XTS_POLAK_RIBIERE => Method::polak_ribiere(),
        xts_method_t::XTS_FLETCHER_REEVES => {
            Method::nlcg(crate::Conjugacy::FletcherReeves)
        }
        xts_method_t::XTS_BFGS => Method::Bfgs,
        xts_method_t::XTS_LBFGS => Method::Lbfgs {
            memory: if memory == 0 { 10 } else { memory },
        },
        xts_method_t::XTS_SR1 => Method::Sr1,
        xts_method_t::XTS_ADAM => Method::adam(),
        xts_method_t::XTS_STEEPEST => Method::Steepest,
        xts_method_t::XTS_SR2 => Method::Sr2,
        xts_method_t::XTS_PSO => Method::pso(),
        xts_method_t::XTS_HESTENES_STIEFEL => {
            Method::nlcg(crate::Conjugacy::HestenesStiefel)
        }
        xts_method_t::XTS_DAI_YUAN => Method::nlcg(crate::Conjugacy::DaiYuan),
        xts_method_t::XTS_CONJUGATE_DESCENT => {
            Method::nlcg(crate::Conjugacy::ConjugateDescent)
        }
        xts_method_t::XTS_HAGER_ZHANG => Method::nlcg(crate::Conjugacy::HagerZhang),
        xts_method_t::XTS_LIU_STOREY => Method::nlcg(crate::Conjugacy::LiuStorey),
        xts_method_t::XTS_FR_PR => Method::nlcg(crate::Conjugacy::FrPr),
        xts_method_t::XTS_NEWTON => Method::Newton {
            kind: NewtonKind::Shifted,
        },
        xts_method_t::XTS_RFO => Method::Newton {
            kind: NewtonKind::Rfo,
        },
    }
}

/// Borrow a 1-D CPU f64 buffer as a DLPack tensor. Caller must
/// [`xts_tensor_free`] it. The buffer must outlive the tensor.
///
/// # Safety
/// `data` points to `len` writable f64s.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn xts_tensor_borrow_cpu_f64(
    data: *mut f64,
    len: usize,
) -> *mut DLManagedTensorVersioned {
    if data.is_null() || len == 0 {
        set_last_error("xts_tensor_borrow_cpu_f64: null or empty");
        return std::ptr::null_mut();
    }
    unsafe { create_borrowed_f64_1d(data, len, DLDeviceType::kDLCPU, 0) }
}

/// Release a tensor created by [`xts_tensor_borrow_cpu_f64`].
///
/// # Safety
/// `tensor` is null or a pointer from this crate's borrow helper.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn xts_tensor_free(tensor: *mut DLManagedTensorVersioned) {
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
) -> Result<&'a [f64], xts_status_t> {
    if t.is_null() {
        set_last_error(&format!("{name}: null tensor"));
        return Err(xts_status_t::XTS_INVALID_PARAMETER);
    }
    let t = unsafe { &*t };
    let dl = &t.dl_tensor;
    if dl.device.device_type != DLDeviceType::kDLCPU {
        set_last_error(&format!(
            "{name}: device {:?} not supported in this build (CPU only)",
            dl.device.device_type as i32
        ));
        return Err(xts_status_t::XTS_UNSUPPORTED_DEVICE);
    }
    if dl.ndim != 1
        || dl.dtype.code != DLDataTypeCode::kDLFloat
        || dl.dtype.bits != 64
        || dl.dtype.lanes != 1
        || dl.shape.is_null()
        || dl.data.is_null()
    {
        set_last_error(&format!("{name}: need rank-1 f64 contiguous DLPack"));
        return Err(xts_status_t::XTS_INVALID_PARAMETER);
    }
    let n = unsafe { *dl.shape as usize };
    if n == 0 {
        set_last_error(&format!("{name}: empty"));
        return Err(xts_status_t::XTS_INVALID_PARAMETER);
    }
    if !dl.strides.is_null() && unsafe { *dl.strides } != 1 {
        set_last_error(&format!("{name}: non-unit stride"));
        return Err(xts_status_t::XTS_INVALID_PARAMETER);
    }
    let ptr = unsafe { (dl.data as *const u8).add(dl.byte_offset as usize) as *const f64 };
    Ok(unsafe { slice::from_raw_parts(ptr, n) })
}

fn cpu_f64_slice_mut<'a>(
    t: *mut DLManagedTensorVersioned,
    name: &str,
) -> Result<&'a mut [f64], xts_status_t> {
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
/// [`xts_status_t::XTS_UNSUPPORTED_DEVICE`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn xts_minimize(
    eval: Option<xts_eval_fn>,
    grad: Option<xts_grad_fn>,
    user: *mut c_void,
    x: *mut DLManagedTensorVersioned,
    ctrl: *const xts_control_t,
    method: xts_method_t,
    out: *mut xts_report_t,
) -> xts_status_t {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let eval = match eval {
            Some(f) => f,
            None => {
                set_last_error("xts_minimize: eval is NULL");
                return xts_status_t::XTS_INVALID_PARAMETER;
            }
        };
        let grad = match grad {
            Some(f) => f,
            None => {
                set_last_error("xts_minimize: grad is NULL");
                return xts_status_t::XTS_INVALID_PARAMETER;
            }
        };
        if matches!(method, xts_method_t::XTS_NEWTON | xts_method_t::XTS_RFO) {
            set_last_error("xts_minimize: Newton/RFO needs xts_minimize_hess");
            return xts_status_t::XTS_INVALID_PARAMETER;
        }
        if ctrl.is_null() || out.is_null() {
            set_last_error("xts_minimize: ctrl/out null");
            return xts_status_t::XTS_INVALID_PARAMETER;
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
            let eval_fn: xts_eval_fn = unsafe { std::mem::transmute(eval_ptr) };
            let grad_fn: xts_grad_fn = unsafe { std::mem::transmute(grad_ptr) };
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
                xts_tensor_free(xt);
                xts_tensor_free(gt);
            }
            if ev_st != xts_status_t::XTS_SUCCESS
                || gr_st != xts_status_t::XTS_SUCCESS
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
            maxmove: if c.maxmove > 0.0 { Some(c.maxmove) } else { None },
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
                    *out = xts_report_t {
                        value: rep.value,
                        steps: rep.steps,
                        grad_norm: rep.grad_norm,
                    };
                }
                xts_status_t::XTS_SUCCESS
            }
            Err(e) => {
                set_last_error(&e.to_string());
                xts_status_t::XTS_INVALID_PARAMETER
            }
        }
    })) {
        Ok(s) => s,
        Err(_) => {
            set_last_error("xts_minimize: panic");
            xts_status_t::XTS_INTERNAL_ERROR
        }
    }
}

/// Minimize with a Newton / RFO direction. `hess` writes a length-`n²`
/// row-major Hessian at the current `x`.
///
/// # Safety
///
/// Same as [`xts_minimize`]. `hess` must be callable for the lifetime
/// of this call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn xts_minimize_hess(
    eval: Option<xts_eval_fn>,
    grad: Option<xts_grad_fn>,
    hess: Option<xts_hess_fn>,
    user: *mut c_void,
    x: *mut DLManagedTensorVersioned,
    ctrl: *const xts_control_t,
    method: xts_method_t,
    out: *mut xts_report_t,
) -> xts_status_t {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let eval = match eval {
            Some(f) => f,
            None => {
                set_last_error("xts_minimize_hess: eval is NULL");
                return xts_status_t::XTS_INVALID_PARAMETER;
            }
        };
        let grad = match grad {
            Some(f) => f,
            None => {
                set_last_error("xts_minimize_hess: grad is NULL");
                return xts_status_t::XTS_INVALID_PARAMETER;
            }
        };
        let hess = match hess {
            Some(f) => f,
            None => {
                set_last_error("xts_minimize_hess: hess is NULL");
                return xts_status_t::XTS_INVALID_PARAMETER;
            }
        };
        if ctrl.is_null() || out.is_null() {
            set_last_error("xts_minimize_hess: ctrl/out null");
            return xts_status_t::XTS_INVALID_PARAMETER;
        }
        let init = match cpu_f64_slice_mut(x, "x") {
            Ok(s) => s.to_vec(),
            Err(st) => return st,
        };
        let n = init.len();
        let eval_ptr = eval as usize;
        let grad_ptr = grad as usize;
        let hess_ptr = hess as usize;
        let user_addr = user as usize;
        let obj = HessianOracle::unbounded(
            n,
            move |xv| {
                let eval_fn: xts_eval_fn = unsafe { std::mem::transmute(eval_ptr) };
                let grad_fn: xts_grad_fn = unsafe { std::mem::transmute(grad_ptr) };
                let user = user_addr as *mut c_void;
                let mut xs = xv.to_vec();
                let xt = unsafe {
                    create_borrowed_f64_1d(
                        xs.as_mut_ptr(),
                        xs.len(),
                        DLDeviceType::kDLCPU,
                        0,
                    )
                };
                let mut value = 0.0;
                let ev_st = unsafe { eval_fn(user, xt, &mut value) };
                let mut g = vec![0.0; xs.len()];
                let gt = unsafe {
                    create_borrowed_f64_1d(g.as_mut_ptr(), g.len(), DLDeviceType::kDLCPU, 0)
                };
                let gr_st = unsafe { grad_fn(user, xt, gt) };
                unsafe {
                    xts_tensor_free(xt);
                    xts_tensor_free(gt);
                }
                if ev_st != xts_status_t::XTS_SUCCESS
                    || gr_st != xts_status_t::XTS_SUCCESS
                {
                    return (f64::INFINITY, Array1::from(g));
                }
                (value, Array1::from(g))
            },
            move |xv| {
                let hess_fn: xts_hess_fn = unsafe { std::mem::transmute(hess_ptr) };
                let user = user_addr as *mut c_void;
                let mut xs = xv.to_vec();
                let xt = unsafe {
                    create_borrowed_f64_1d(
                        xs.as_mut_ptr(),
                        xs.len(),
                        DLDeviceType::kDLCPU,
                        0,
                    )
                };
                let mut h = vec![0.0; n * n];
                let ht = unsafe {
                    create_borrowed_f64_1d(h.as_mut_ptr(), h.len(), DLDeviceType::kDLCPU, 0)
                };
                let st = unsafe { hess_fn(user, xt, ht) };
                unsafe {
                    xts_tensor_free(xt);
                    xts_tensor_free(ht);
                }
                if st != xts_status_t::XTS_SUCCESS {
                    return Array2::eye(n);
                }
                Array2::from_shape_vec((n, n), h).unwrap_or_else(|_| Array2::eye(n))
            },
        );
        let c = unsafe { &*ctrl };
        let control = Control {
            maxiter: c.maxiter,
            gtol: c.gtol,
            istep: if c.istep > 0.0 { c.istep } else { 1.0 },
            maxmove: if c.maxmove > 0.0 { Some(c.maxmove) } else { None },
        };
        let kind = match method {
            xts_method_t::XTS_RFO => NewtonKind::Rfo,
            _ => NewtonKind::Shifted,
        };
        match minimize_newton(&obj, Array1::from(init), &control, kind) {
            Ok(rep) => {
                let dest = match cpu_f64_slice_mut(x, "x") {
                    Ok(s) => s,
                    Err(st) => return st,
                };
                dest.copy_from_slice(rep.coords.as_slice().expect("contiguous"));
                unsafe {
                    *out = xts_report_t {
                        value: rep.value,
                        steps: rep.steps,
                        grad_norm: rep.grad_norm,
                    };
                }
                xts_status_t::XTS_SUCCESS
            }
            Err(e) => {
                set_last_error(&e.to_string());
                xts_status_t::XTS_INVALID_PARAMETER
            }
        }
    })) {
        Ok(s) => s,
        Err(_) => {
            set_last_error("xts_minimize_hess: panic");
            xts_status_t::XTS_INTERNAL_ERROR
        }
    }
}

/// Minimize an eindir-compatible objective without taking ownership of it.
///
/// The objective must provide an analytic gradient and the producer's ABI
/// stamp must be compatible with this build. The objective remains owned by
/// the caller for the full duration of the call.
///
/// # Safety
///
/// `objective`, `stamp`, `x`, `ctrl`, and `out` must be valid for the duration
/// of this call. `objective` and `stamp` must come from the same compatible
/// eindir ABI family. `x` must be a writable rank-1 CPU f64 tensor.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn xts_minimize_eindir(
    objective: *const eindir_objective_t,
    stamp: *const eindir_abi_stamp_t,
    x: *mut DLManagedTensorVersioned,
    ctrl: *const xts_control_t,
    method: xts_method_t,
    out: *mut xts_report_t,
) -> xts_status_t {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        if objective.is_null() || stamp.is_null() || ctrl.is_null() || out.is_null() {
            set_last_error("xts_minimize_eindir: null argument");
            return xts_status_t::XTS_INVALID_PARAMETER;
        }
        if unsafe { eindir_core_abi_compatible(stamp) } == 0 {
            set_last_error("xts_minimize_eindir: incompatible eindir ABI stamp");
            return xts_status_t::XTS_INVALID_PARAMETER;
        }
        if unsafe { eindir_objective_has_grad(objective) } == 0 {
            set_last_error("xts_minimize_eindir: objective has no gradient");
            return xts_status_t::XTS_INVALID_PARAMETER;
        }
        let init = match cpu_f64_slice_mut(x, "x") {
            Ok(s) => s.to_vec(),
            Err(st) => return st,
        };
        let dim = unsafe { (*objective).dim };
        if dim != init.len() {
            set_last_error(&format!(
                "xts_minimize_eindir: x length {} != objective dim {}",
                init.len(),
                dim
            ));
            return xts_status_t::XTS_INVALID_PARAMETER;
        }
        let objective_addr = objective as usize;
        let obj = Oracle::unbounded(dim, move |xv| {
            let objective = objective_addr as *const eindir_objective_t;
            let mut xs = xv.to_vec();
            let xt = unsafe {
                create_borrowed_f64_1d(xs.as_mut_ptr(), xs.len(), DLDeviceType::kDLCPU, 0)
            };
            let mut value = 0.0;
            let eval_status = unsafe { eindir_objective_eval(objective, xt, &mut value) };
            let mut gradient = vec![0.0; xs.len()];
            let gt = unsafe {
                create_borrowed_f64_1d(
                    gradient.as_mut_ptr(),
                    gradient.len(),
                    DLDeviceType::kDLCPU,
                    0,
                )
            };
            let grad_status = unsafe { eindir_objective_grad(objective, xt, gt) };
            unsafe {
                xts_tensor_free(xt);
                xts_tensor_free(gt);
            }
            if eval_status != eindir_status_t::EINDIR_SUCCESS
                || grad_status != eindir_status_t::EINDIR_SUCCESS
            {
                set_last_error("xts_minimize_eindir: eindir evaluation failed");
                return (f64::INFINITY, Array1::from(gradient));
            }
            (value, Array1::from(gradient))
        });
        let c = unsafe { &*ctrl };
        let control = Control {
            maxiter: c.maxiter,
            gtol: c.gtol,
            istep: if c.istep > 0.0 { c.istep } else { 1.0 },
            maxmove: if c.maxmove > 0.0 { Some(c.maxmove) } else { None },
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
                    *out = xts_report_t {
                        value: rep.value,
                        steps: rep.steps,
                        grad_norm: rep.grad_norm,
                    };
                }
                xts_status_t::XTS_SUCCESS
            }
            Err(e) => {
                set_last_error(&e.to_string());
                xts_status_t::XTS_INVALID_PARAMETER
            }
        }
    })) {
        Ok(s) => s,
        Err(_) => {
            set_last_error("xts_minimize_eindir: panic");
            xts_status_t::XTS_INTERNAL_ERROR
        }
    }
}

#[cfg(test)]
mod device_tests {
    use super::*;

    #[test]
    fn cuda_tag_is_unsupported() {
        let mut buf = [0.0_f64; 2];
        let t = unsafe { create_borrowed_f64_1d(buf.as_mut_ptr(), 2, DLDeviceType::kDLCUDA, 0) };
        let err = cpu_f64_slice(t, "cuda").unwrap_err();
        unsafe { xts_tensor_free(t) };
        assert_eq!(err, xts_status_t::XTS_UNSUPPORTED_DEVICE);
    }
}
