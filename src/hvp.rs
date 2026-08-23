//! Matrix-free Newton: Hessian action instead of a Hessian matrix.
//!
//! The dense path in [`crate::newton`] factors an explicit `n x n`
//! Hessian, which prices every step at O(n^3) flops and O(n^2) memory.
//! TAO's BNK family never forms the matrix: it asks the objective for
//! the action `H(x) v` and runs Steihaug-Toint conjugate gradients
//! inside a trust region, so cost scales with the number of actions,
//! not with n^3. This module is that path for eindir objectives.
//!
//! Nocedal and Wright, *Numerical Optimization*, algorithms 4.1 and
//! 7.2, <https://doi.org/10.1007/978-0-387-40065-5>.
//! Steihaug, *The Conjugate Gradient Method and Trust Regions in Large
//! Scale Optimization*, <https://doi.org/10.1137/0720042>.

use eindir_core::DifferentiableObjective;
use ndarray::{Array1, ArrayView1};

use crate::control::Control;
use crate::error::{Error, Result};
use crate::report::Report;
use crate::scg::DirectionalCurvature;
use crate::trust::{reduction_ratio, update_radius};
use crate::vecops::{axpy, dot, nrm2};

/// Hessian action `H(x) v` without forming the matrix.
pub trait HessianVector: DifferentiableObjective<f64> {
    /// `grad^2 f(x) . v`.
    fn hessian_vector(&self, x: ArrayView1<f64>, v: ArrayView1<f64>) -> Array1<f64>;
}

/// Every Hessian action supplies a directional curvature for free:
/// `d . H d` is one action and one dot, so any [`HessianVector`]
/// objective can drive [`crate::minimize_scg_exact`] as well.
impl<T: HessianVector + ?Sized> DirectionalCurvature for T {
    fn directional_curvature(&self, x: ArrayView1<f64>, d: ArrayView1<f64>) -> Option<f64> {
        let hd = self.hessian_vector(x, d);
        Some(dot(d, hd.view()))
    }
}

/// Closure adapter: fused `(f, g)` plus a Hessian action.
pub struct HvpOracle<F, H>
where
    F: Fn(ArrayView1<f64>) -> (f64, Array1<f64>) + Send + Sync,
    H: Fn(ArrayView1<f64>, ArrayView1<f64>) -> Array1<f64> + Send + Sync,
{
    f: F,
    hvp: H,
    bounds: eindir_core::Bounds<f64>,
}

impl<F, H> HvpOracle<F, H>
where
    F: Fn(ArrayView1<f64>) -> (f64, Array1<f64>) + Send + Sync,
    H: Fn(ArrayView1<f64>, ArrayView1<f64>) -> Array1<f64> + Send + Sync,
{
    /// Wide-box oracle of dimension `dim`.
    pub fn unbounded(dim: usize, f: F, hvp: H) -> Self {
        const LO: f64 = -1e12;
        const HI: f64 = 1e12;
        Self {
            f,
            hvp,
            bounds: eindir_core::Bounds::new(
                Array1::from_elem(dim, LO),
                Array1::from_elem(dim, HI),
                0.0,
            ),
        }
    }
}

impl<F, H> eindir_core::Objective<f64> for HvpOracle<F, H>
where
    F: Fn(ArrayView1<f64>) -> (f64, Array1<f64>) + Send + Sync,
    H: Fn(ArrayView1<f64>, ArrayView1<f64>) -> Array1<f64> + Send + Sync,
{
    fn dim(&self) -> usize {
        self.bounds.dims
    }

    fn bounds(&self) -> &eindir_core::Bounds<f64> {
        &self.bounds
    }

    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        (self.f)(x).0
    }
}

impl<F, H> eindir_core::Gradient<f64> for HvpOracle<F, H>
where
    F: Fn(ArrayView1<f64>) -> (f64, Array1<f64>) + Send + Sync,
    H: Fn(ArrayView1<f64>, ArrayView1<f64>) -> Array1<f64> + Send + Sync,
{
    fn dim(&self) -> usize {
        self.bounds.dims
    }

    fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
        (self.f)(x).1
    }
}

impl<F, H> DifferentiableObjective<f64> for HvpOracle<F, H>
where
    F: Fn(ArrayView1<f64>) -> (f64, Array1<f64>) + Send + Sync,
    H: Fn(ArrayView1<f64>, ArrayView1<f64>) -> Array1<f64> + Send + Sync,
{
    fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        (self.f)(x)
    }
}

impl<F, H> HessianVector for HvpOracle<F, H>
where
    F: Fn(ArrayView1<f64>) -> (f64, Array1<f64>) + Send + Sync,
    H: Fn(ArrayView1<f64>, ArrayView1<f64>) -> Array1<f64> + Send + Sync,
{
    fn hessian_vector(&self, x: ArrayView1<f64>, v: ArrayView1<f64>) -> Array1<f64> {
        (self.hvp)(x, v)
    }
}

/// A Hessian action from central gradient differences, for objectives
/// with no analytic action: `H v ~ (g(x + e v) - g(x - e v)) / 2e`.
/// Two gradient calls per action.
pub struct FdHvp<'a, O: DifferentiableObjective<f64> + ?Sized> {
    obj: &'a O,
    eps: f64,
}

impl<'a, O: DifferentiableObjective<f64> + ?Sized> FdHvp<'a, O> {
    /// Wrap `obj`; `eps` scales the probe (1e-6 suits doubles at unit
    /// curvature).
    pub fn new(obj: &'a O, eps: f64) -> Self {
        Self { obj, eps }
    }
}

impl<'a, O: DifferentiableObjective<f64> + ?Sized> eindir_core::Objective<f64> for FdHvp<'a, O> {
    fn dim(&self) -> usize {
        eindir_core::Objective::dim(self.obj)
    }
    fn bounds(&self) -> &eindir_core::Bounds<f64> {
        self.obj.bounds()
    }
    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        self.obj.eval(x)
    }
}

impl<'a, O: DifferentiableObjective<f64> + ?Sized> eindir_core::Gradient<f64> for FdHvp<'a, O> {
    fn dim(&self) -> usize {
        eindir_core::Objective::dim(self.obj)
    }
    fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
        self.obj.value_and_gradient(x).1
    }
}

impl<'a, O: DifferentiableObjective<f64> + ?Sized> DifferentiableObjective<f64> for FdHvp<'a, O> {
    fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        self.obj.value_and_gradient(x)
    }
}

impl<'a, O: DifferentiableObjective<f64> + ?Sized> HessianVector for FdHvp<'a, O> {
    fn hessian_vector(&self, x: ArrayView1<f64>, v: ArrayView1<f64>) -> Array1<f64> {
        let vn = nrm2(v);
        if vn <= 0.0 {
            return Array1::zeros(v.len());
        }
        let h = self.eps / vn;
        let mut xp = x.to_owned();
        axpy(h, v, &mut xp);
        let mut xm = x.to_owned();
        axpy(-h, v, &mut xm);
        let gp = self.obj.value_and_gradient(xp.view()).1;
        let gm = self.obj.value_and_gradient(xm.view()).1;
        (gp - gm) / (2.0 * h)
    }
}

/// Steihaug-Toint CG on the trust-region subproblem: approximately
/// minimize `g.p + p.Hp/2` subject to `||p|| <= radius`. Returns the
/// step and the model reduction `m(0) - m(p)`; every Hessian access is
/// one action.
pub fn steihaug_cg<O>(
    obj: &O,
    x: ArrayView1<f64>,
    grad: &Array1<f64>,
    radius: f64,
    rtol: f64,
    maxiter: usize,
) -> (Array1<f64>, f64)
where
    O: HessianVector + ?Sized,
{
    let n = grad.len();
    let mut z = Array1::<f64>::zeros(n);
    let mut hz = Array1::<f64>::zeros(n);
    let mut r = grad.clone();
    let mut d = grad.mapv(|v| -v);
    let mut rr = dot(r.view(), r.view());
    let gnorm = rr.sqrt();
    let stop = (rtol * gnorm).max(f64::MIN_POSITIVE);

    let model_drop = |p: &Array1<f64>, hp: &Array1<f64>| {
        -(dot(grad.view(), p.view()) + 0.5 * dot(p.view(), hp.view()))
    };

    for _ in 0..maxiter {
        let hd = obj.hessian_vector(x, d.view());
        let dhd = dot(d.view(), hd.view());
        if !dhd.is_finite() {
            // A broken action cannot price curvature; take the
            // steepest boundary step and let the ratio test judge it.
            let p = grad.mapv(|v| -radius * v / gnorm.max(f64::MIN_POSITIVE));
            let hp = obj.hessian_vector(x, p.view());
            let drop = model_drop(&p, &hp);
            return (p, drop);
        }
        if dhd <= 0.0 {
            // Negative curvature: follow d to the boundary.
            let tau = boundary_tau(&z, &d, radius);
            axpy(tau, d.view(), &mut z);
            axpy(tau, hd.view(), &mut hz);
            let drop = model_drop(&z, &hz);
            return (z, drop);
        }
        let alpha = rr / dhd;
        let mut z_next = z.clone();
        axpy(alpha, d.view(), &mut z_next);
        if nrm2(z_next.view()) >= radius {
            let tau = boundary_tau(&z, &d, radius);
            axpy(tau, d.view(), &mut z);
            axpy(tau, hd.view(), &mut hz);
            let drop = model_drop(&z, &hz);
            return (z, drop);
        }
        z = z_next;
        axpy(alpha, hd.view(), &mut hz);
        axpy(alpha, hd.view(), &mut r);
        let rr_next = dot(r.view(), r.view());
        if rr_next.sqrt() < stop {
            break;
        }
        let beta = rr_next / rr;
        rr = rr_next;
        for (di, ri) in d.iter_mut().zip(r.iter()) {
            *di = -ri + beta * *di;
        }
    }
    let drop = model_drop(&z, &hz);
    (z, drop)
}

/// Positive `tau` with `||z + tau d|| = radius`.
fn boundary_tau(z: &Array1<f64>, d: &Array1<f64>, radius: f64) -> f64 {
    let dd = dot(d.view(), d.view());
    if dd <= 0.0 {
        return 0.0;
    }
    let zd = dot(z.view(), d.view());
    let zz = dot(z.view(), z.view());
    let disc = (zd * zd + dd * (radius * radius - zz)).max(0.0);
    (-zd + disc.sqrt()) / dd
}

const ETA_ACCEPT: f64 = 0.1;
const RADIUS_FLOOR: f64 = 1e-14;

/// Matrix-free Newton: Steihaug-Toint CG inside a Nocedal-Wright trust
/// region. `control.maxmove` caps the trust radius, so the region
/// plays the role the step clip plays elsewhere; `control.istep` seeds
/// the initial radius when positive.
pub fn minimize_newton_cg<O>(obj: &O, init: impl Into<Array1<f64>>, control: &Control) -> Result<Report>
where
    O: HessianVector + ?Sized,
{
    let mut pos = init.into();
    if pos.len() != eindir_core::Objective::dim(obj) {
        return Err(Error::Dim {
            got: pos.len(),
            dim: eindir_core::Objective::dim(obj),
        });
    }
    pos = obj.bounds().clip(pos.view());
    let (mut value, mut grad) = obj.value_and_gradient(pos.view());
    let n = pos.len();

    let rmax = control.maxmove.unwrap_or(f64::INFINITY).max(RADIUS_FLOOR);
    let mut radius = if control.istep > 0.0 {
        control.istep.min(rmax)
    } else {
        1.0_f64.min(rmax)
    };

    for step in 0..control.maxiter {
        let gnorm = nrm2(grad.view());
        if !gnorm.is_finite() {
            return Err(Error::TrustCollapsed { steps: step });
        }
        if gnorm < control.gtol {
            return Ok(Report {
                value,
                coords: pos,
                steps: step,
                grad_norm: gnorm,
            });
        }
        // Inexact-Newton forcing keeps early CG cheap and the tail
        // superlinear (Nocedal-Wright eq 7.3).
        let rtol = gnorm.sqrt().min(0.5);
        let (p, pred) = steihaug_cg(obj, pos.view(), &grad, radius, rtol, 2 * n);
        let pnorm = nrm2(p.view());
        if pnorm <= 0.0 || pred <= 0.0 {
            radius *= 0.25;
            if radius < RADIUS_FLOOR {
                return Err(Error::TrustCollapsed { steps: step });
            }
            continue;
        }
        let mut trial = pos.clone();
        axpy(1.0, p.view(), &mut trial);
        let trial = obj.bounds().clip(trial.view());
        let (ft, gt) = obj.value_and_gradient(trial.view());
        let rho = if ft.is_finite() {
            reduction_ratio(value - ft, pred)
        } else {
            -1.0
        };
        radius = update_radius(radius, rho, pnorm, rmax);
        if radius < RADIUS_FLOOR {
            return Err(Error::TrustCollapsed { steps: step });
        }
        if rho > ETA_ACCEPT {
            pos = trial;
            value = ft;
            grad = gt;
        }
    }
    Ok(Report {
        value,
        coords: pos,
        steps: control.maxiter,
        grad_norm: nrm2(grad.view()),
    })
}
