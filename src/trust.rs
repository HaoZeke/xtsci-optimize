//! Trust-region steps on a dense host Hessian.
//!
//! Two distinct constructions live here:
//!
//! - Powell dogleg (Nocedal and Wright, algorithm 4.1): interpolate
//!   the Cauchy and Newton points inside `||p|| <= delta`.
//! - Sella [`TrustRegion`]: `cons(s) = ||s||`, target `delta`, solved
//!   on the Baker / Sella quasi-Newton family `s(α)`. This is not
//!   IRCTrustRegion (`||(s + d1) * sqrt(m)||`).
//!
//! Nocedal and Wright, *Numerical Optimization*, algorithm 4.1,
//! <https://doi.org/10.1007/978-0-387-40065-5>.
//! Dennis and Schnabel, *Numerical Methods for Unconstrained
//! Optimization and Nonlinear Equations*,
//! <https://doi.org/10.1137/1.9781611971200>.
//! Baker, *An algorithm for the location of transition states*,
//! <https://doi.org/10.1002/jcc.540070402>.

use ndarray::{Array1, Array2, ArrayView1};

use crate::error::{Error, Result};
use crate::newton::shifted_newton;
use crate::step::l2;
use crate::vecops::{axpy, dot, nrm2};

const RHO_BAD: f64 = 0.25;
const RHO_GOOD: f64 = 0.75;
const BOUNDARY: f64 = 0.8;

/// Cauchy / Newton dogleg point inside radius `delta`.
pub fn dogleg_direction(hess: &Array2<f64>, grad: &Array1<f64>, delta: f64) -> Array1<f64> {
    let radius = delta.max(1e-16);
    let p_b = shifted_newton(hess, grad);
    let nb = l2(&p_b);
    if nb <= radius {
        return p_b;
    }
    let hg = hess.dot(grad);
    let ghg = grad.dot(&hg);
    let gg = grad.dot(grad);
    if gg <= 0.0 {
        return Array1::zeros(grad.len());
    }
    let p_u = if ghg > 1e-16 {
        grad.mapv(|v| -(gg / ghg) * v)
    } else {
        let ng = gg.sqrt();
        return grad.mapv(|v| -radius * v / ng);
    };
    let nu = l2(&p_u);
    if nu >= radius {
        return p_u.mapv(|v| v * (radius / nu));
    }
    let diff = &p_b - &p_u;
    let a = diff.dot(&diff);
    if a <= 1e-32 {
        return p_u;
    }
    let b = 2.0 * p_u.dot(&diff);
    let c = p_u.dot(&p_u) - radius * radius;
    let disc = (b * b - 4.0 * a * c).max(0.0);
    let tau = ((-b + disc.sqrt()) / (2.0 * a)).clamp(0.0, 1.0);
    &p_u + &(&diff * tau)
}

/// Predicted reduction `-g·p - 1/2 p·H p`.
pub fn predicted_reduction(hess: &Array2<f64>, grad: &Array1<f64>, p: &Array1<f64>) -> f64 {
    let hp = hess.dot(p);
    -grad.dot(p) - 0.5 * p.dot(&hp)
}

/// Trust-region ratio `ared / pred`.
pub fn reduction_ratio(ared: f64, pred: f64) -> f64 {
    if pred.abs() <= 1e-16 {
        if ared >= 0.0 { 1.0 } else { -1.0 }
    } else {
        ared / pred
    }
}

/// Nocedal-Wright radius update. Returns the new radius.
pub fn update_radius(radius: f64, rho: f64, pnorm: f64, rmax: f64) -> f64 {
    let r = radius.max(1e-16);
    if rho < RHO_BAD {
        (0.25 * r).max(1e-16)
    } else if rho > RHO_GOOD && pnorm >= BOUNDARY * r {
        (2.0 * r).min(rmax.max(r))
    } else {
        r
    }
}

/// True when the trial point is accepted (`ρ > 0`).
pub fn accept_ratio(rho: f64) -> bool {
    rho > 0.0
}

const DENOM_FLOOR: f64 = 1e-16;
const CONS_FLOOR: f64 = 1e-12;

/// Sella `TrustRegion`: `cons(s) = ||s||`, target radius `delta`.
///
/// Distinct from IRCTrustRegion, which constrains
/// `||(s + d1) * sqrt(m)||`. This type never mass-weights the step
/// and never shifts it by an IRC tangent.
#[derive(Clone, Debug)]
pub struct TrustRegion {
    /// Trust radius. The accepted step satisfies `||s|| <= delta`.
    pub delta: f64,
    /// Number of Hessian modes whose curvature sign is flipped
    /// (0 = minimum, 1 = first-order saddle).
    pub order: usize,
    /// Absolute tolerance on `||s|| - delta`.
    pub tol: f64,
    /// Newton / bisection iterations.
    pub maxiter: usize,
}

/// Accepted restricted step.
///
/// `cons` is `||s||` when the unconstrained QN step is inside the
/// region, and `delta` when the solver sat on the bound.
#[derive(Clone, Debug)]
pub struct RestrictedStep {
    /// Displacement `s`.
    pub step: Array1<f64>,
    /// Constraint value Sella `get_s` reports: `||s||` or `delta`.
    pub cons: f64,
}

impl RestrictedStep {
    /// Euclidean length of the step (vecops `nrm2`).
    pub fn nrm2(&self) -> f64 {
        nrm2(self.step.view())
    }
}

impl TrustRegion {
    /// Radius `delta`, minimum-mode QN (`order = 0`).
    pub fn new(delta: f64) -> Self {
        Self {
            delta: delta.max(0.0),
            order: 0,
            tol: 1e-10,
            maxiter: 1000,
        }
    }

    /// Flip the lowest `order` Hessian modes, Sella `order`.
    pub fn with_order(mut self, order: usize) -> Self {
        self.order = order;
        self
    }

    /// Restrict the Sella quasi-Newton family `s(α)` to the trust sphere.
    ///
    /// `α = 0` is the unconstrained QN step. When that step is longer
    /// than `delta`, a Newton-bisection hybrid solves `||s(α)|| = delta`
    /// for `α > 0` (Baker 1986 / Sella `TrustRegion.get_s`).
    pub fn restrict_qn(&self, hess: &Array2<f64>, grad: &Array1<f64>) -> Result<RestrictedStep> {
        let family = QnFamily::new(hess, grad, self.order)?;
        let (step, cons) = restrict(
            |alpha| {
                let (s, dsda) = family.get_s(alpha);
                let (val, dval) = trust_cons(&s, &dsda);
                (s, val, dval)
            },
            RestrictParams {
                alpha0: 0.0,
                alphamin: 0.0,
                alphamax: f64::INFINITY,
                slope: -1.0,
                newton_safe: true,
                delta: self.delta,
                tol: self.tol,
                maxiter: self.maxiter,
            },
        )?;
        Ok(RestrictedStep { step, cons })
    }
}

/// Sella `TrustRegion.cons`: `||s||` and `d||s||/dα = (ds/dα · s) / ||s||`.
fn trust_cons(s: &Array1<f64>, dsda: &Array1<f64>) -> (f64, f64) {
    let val = nrm2(s.view());
    let dval = dot(dsda.view(), s.view()) / val.max(CONS_FLOOR);
    (val, dval)
}

struct RestrictParams {
    alpha0: f64,
    alphamin: f64,
    alphamax: f64,
    slope: f64,
    newton_safe: bool,
    delta: f64,
    tol: f64,
    maxiter: usize,
}

/// Sella `BaseRestrictedStep.get_s` on a one-parameter family.
fn restrict(
    mut eval: impl FnMut(f64) -> (Array1<f64>, f64, f64),
    p: RestrictParams,
) -> Result<(Array1<f64>, f64)> {
    let mut lower = p.alphamin;
    let mut upper = p.alphamax;
    let mut alpha = p.alpha0;
    let (mut s, mut val, mut dval) = eval(alpha);
    if !val.is_finite() {
        return Err(Error::RestrictedStep);
    }
    if val < p.delta {
        return Ok((s, val));
    }
    let mut err = val - p.delta;
    for niter in 0..p.maxiter {
        if err.abs() <= p.tol {
            return Ok((s, p.delta));
        }
        if lower.next_up() >= upper {
            return Ok((s, p.delta));
        }
        if err * p.slope > 0.0 {
            upper = alpha;
        } else {
            lower = alpha;
        }
        let a1 = alpha - err / dval;
        alpha = if !a1.is_finite() || a1 <= lower || a1 >= upper || (niter > 4 && !p.newton_safe) {
            let a2 = 0.5 * (lower + upper);
            if a2.is_infinite() {
                alpha + 1.0_f64.max(0.5 * alpha) * a2.signum()
            } else {
                a2
            }
        } else {
            a1
        };
        let next = eval(alpha);
        s = next.0;
        val = next.1;
        dval = next.2;
        if !val.is_finite() {
            return Err(Error::RestrictedStep);
        }
        err = val - p.delta;
    }
    Err(Error::RestrictedStep)
}

/// Sella `QuasiNewton`: `s(α) = -V (Vg / (L + α ones))`.
struct QnFamily {
    shift: Array1<f64>,
    ones: Array1<f64>,
    evecs: Array2<f64>,
    vg: Array1<f64>,
}

impl QnFamily {
    fn new(hess: &Array2<f64>, grad: &Array1<f64>, order: usize) -> Result<Self> {
        let n = grad.len();
        if hess.nrows() != n || hess.ncols() != n {
            return Err(Error::Dim {
                got: hess.nrows(),
                dim: n,
            });
        }
        let (evals, evecs) = sym_eig_ascending(hess);
        let order = order.min(n);
        let mut shift = Array1::zeros(n);
        let mut ones = Array1::zeros(n);
        for i in 0..n {
            let a = evals[i].abs();
            if i < order {
                shift[i] = -a;
                ones[i] = -1.0;
            } else {
                shift[i] = a;
                ones[i] = 1.0;
            }
        }
        let vg = gemv_t(&evecs, grad.view());
        Ok(Self {
            shift,
            ones,
            evecs,
            vg,
        })
    }

    fn get_s(&self, alpha: f64) -> (Array1<f64>, Array1<f64>) {
        let n = self.vg.len();
        let mut sproj = Array1::zeros(n);
        let mut dproj = Array1::zeros(n);
        for i in 0..n {
            let raw = self.shift[i] + alpha * self.ones[i];
            let denom = if raw.abs() < DENOM_FLOOR {
                DENOM_FLOOR.copysign(if raw == 0.0 { 1.0 } else { raw })
            } else {
                raw
            };
            sproj[i] = self.vg[i] / denom;
            dproj[i] = sproj[i] * self.ones[i] / denom;
        }
        let mut step = gemv(&self.evecs, sproj.view());
        for v in step.iter_mut() {
            *v = -*v;
        }
        let dsda = gemv(&self.evecs, dproj.view());
        (step, dsda)
    }
}

fn gemv(a: &Array2<f64>, x: ArrayView1<f64>) -> Array1<f64> {
    let mut y = Array1::zeros(a.nrows());
    for j in 0..x.len() {
        axpy(x[j], a.column(j), &mut y);
    }
    y
}

fn gemv_t(a: &Array2<f64>, x: ArrayView1<f64>) -> Array1<f64> {
    let mut y = Array1::zeros(a.ncols());
    for j in 0..a.ncols() {
        y[j] = dot(a.column(j), x);
    }
    y
}

fn sym_eig_ascending(a: &Array2<f64>) -> (Array1<f64>, Array2<f64>) {
    let (evals, evecs) = sym_eig_jacobi(a.clone());
    let n = evals.len();
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&i, &j| evals[i].total_cmp(&evals[j]));
    let mut lam = Array1::zeros(n);
    let mut v = Array2::zeros((n, n));
    for (k, &i) in idx.iter().enumerate() {
        lam[k] = evals[i];
        v.column_mut(k).assign(&evecs.column(i));
    }
    (lam, v)
}

/// Cyclic Jacobi eigendecomposition. Eigenvalues are unsorted.
fn sym_eig_jacobi(mut a: Array2<f64>) -> (Array1<f64>, Array2<f64>) {
    let k = a.nrows();
    let mut v = Array2::<f64>::eye(k);
    for _ in 0..64 {
        let mut off = 0.0;
        for p in 0..k {
            for q in (p + 1)..k {
                off += a[(p, q)] * a[(p, q)];
            }
        }
        let scale = a.diag().iter().map(|d| d.abs()).fold(1.0, f64::max);
        if off.sqrt() <= 1e-15 * scale {
            break;
        }
        for p in 0..k {
            for q in (p + 1)..k {
                let apq = a[(p, q)];
                if apq.abs() <= 1e-300 {
                    continue;
                }
                let theta = (a[(q, q)] - a[(p, p)]) / (2.0 * apq);
                let sign = if theta >= 0.0 { 1.0 } else { -1.0 };
                let t = sign / (theta.abs() + (theta * theta + 1.0).sqrt());
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for i in 0..k {
                    let aip = a[(i, p)];
                    let aiq = a[(i, q)];
                    a[(i, p)] = c * aip - s * aiq;
                    a[(i, q)] = s * aip + c * aiq;
                }
                for i in 0..k {
                    let api = a[(p, i)];
                    let aqi = a[(q, i)];
                    a[(p, i)] = c * api - s * aqi;
                    a[(q, i)] = s * api + c * aqi;
                }
                for i in 0..k {
                    let vip = v[(i, p)];
                    let viq = v[(i, q)];
                    v[(i, p)] = c * vip - s * viq;
                    v[(i, q)] = s * vip + c * viq;
                }
            }
        }
    }
    (a.diag().to_owned(), v)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn dogleg_takes_the_newton_point_inside_a_large_radius() {
        let h = Array2::<f64>::eye(2) * 2.0;
        let g = array![2.0, 0.0];
        let p = dogleg_direction(&h, &g, 10.0);
        assert!((p[0] + 1.0).abs() < 1e-10);
        assert!(p[1].abs() < 1e-10);
    }

    #[test]
    fn dogleg_stays_on_the_trust_sphere() {
        let h = Array2::<f64>::eye(2) * 2.0;
        let g = array![2.0, 0.0];
        let p = dogleg_direction(&h, &g, 0.1);
        let n = l2(&p);
        assert!((n - 0.1).abs() < 1e-12);
        assert!(p[0] < 0.0);
    }

    #[test]
    fn qn_step_inside_a_large_radius_is_unconstrained() {
        let h = Array2::<f64>::eye(2) * 2.0;
        let g = array![2.0, 0.0];
        let acc = TrustRegion::new(10.0).restrict_qn(&h, &g).unwrap();
        assert!((acc.step[0] + 1.0).abs() < 1e-10);
        assert!(acc.step[1].abs() < 1e-10);
        assert!(acc.nrm2() <= 10.0);
        assert!(acc.nrm2() < 10.0 - 1e-6);
        assert!((acc.cons - acc.nrm2()).abs() < 1e-12);
    }

    #[test]
    fn qn_step_sits_on_the_bound_when_newton_is_longer() {
        let h = Array2::<f64>::eye(2) * 2.0;
        let g = array![2.0, 0.0];
        let delta = 0.1;
        let acc = TrustRegion::new(delta).restrict_qn(&h, &g).unwrap();
        let n = acc.nrm2();
        assert!(n <= delta + 1e-10);
        assert!((n - delta).abs() < 1e-9);
        assert!((acc.cons - delta).abs() < 1e-14);
        assert!(acc.step[0] < 0.0);
    }

    #[test]
    fn retract_of_the_restricted_step_stays_on_the_trust_ball() {
        use crate::manifold::{Euclidean, Manifold};
        let h = Array2::<f64>::eye(3) * 2.0;
        let g = array![2.0, -1.0, 0.5];
        let delta = 0.2;
        let x = array![1.0, -2.0, 0.5];
        let acc = TrustRegion::new(delta).restrict_qn(&h, &g).unwrap();
        assert!(acc.nrm2() <= delta + 1e-10);
        let y = Euclidean.retract(&x, &acc.step);
        let v = Euclidean.project(&x, &(&y - &x));
        assert!(nrm2(v.view()) <= delta + 1e-10);
        let t = Euclidean.transport(&x, &y, &acc.step);
        assert!((nrm2(t.view()) - acc.nrm2()).abs() < 1e-12);
    }

    #[test]
    fn cons_is_euclidean_norm_not_mass_weighted_irc() {
        let h = Array2::<f64>::eye(2) * 2.0;
        let g = array![2.0, 0.0];
        let acc = TrustRegion::new(0.1).restrict_qn(&h, &g).unwrap();
        assert!((acc.nrm2() - 0.1).abs() < 1e-9);
        let irc = (acc.step[0] * 10.0).hypot(acc.step[1]);
        assert!((irc - 0.1).abs() > 0.5);
    }

    #[test]
    fn saddle_order_flips_the_lowest_mode() {
        let h = array![[-2.0, 0.0], [0.0, 2.0]];
        let g = array![2.0, 2.0];
        let free = TrustRegion::new(10.0)
            .with_order(1)
            .restrict_qn(&h, &g)
            .unwrap();
        assert!((free.step[0] - 1.0).abs() < 1e-8);
        assert!((free.step[1] + 1.0).abs() < 1e-8);
        let bound = TrustRegion::new(0.2)
            .with_order(1)
            .restrict_qn(&h, &g)
            .unwrap();
        assert!(bound.nrm2() <= 0.2 + 1e-10);
        assert!((bound.nrm2() - 0.2).abs() < 1e-8);
    }

    #[test]
    fn a_stationary_point_returns_the_zero_step() {
        let h = Array2::<f64>::eye(2);
        let g = array![0.0, 0.0];
        let acc = TrustRegion::new(0.5).restrict_qn(&h, &g).unwrap();
        assert!(acc.nrm2() < 1e-14);
    }

    #[test]
    fn hessian_dimension_mismatch_is_dim() {
        let h = Array2::<f64>::eye(2);
        let g = array![1.0, 2.0, 3.0];
        match TrustRegion::new(1.0).restrict_qn(&h, &g).unwrap_err() {
            Error::Dim { got: 2, dim: 3 } => {}
            other => panic!("{other:?}"),
        }
    }
}
