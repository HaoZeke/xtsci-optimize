//! Dense BFGS, L-BFGS, inverse SR1, and SR2.

use eindir_core::{DifferentiableObjective, Objective};
use ndarray::{Array1, Array2};

use crate::control::Control;
use crate::error::{Error, Result};
use crate::linesearch::LineSearch;
use crate::report::Report;
use crate::step::{l2, next_istep, take_step};

const CURVATURE: f64 = 1e-12;
const SR1_SKIP: f64 = 1e-8;

/// Inverse-BFGS (Nocedal-Wright 6.17).
pub fn minimize_bfgs<O>(
    obj: &O,
    init: impl Into<Array1<f64>>,
    control: &Control,
    linesearch: LineSearch,
) -> Result<Report>
where
    O: DifferentiableObjective<f64> + ?Sized,
{
    let mut pos = start(obj, init)?;
    let n = pos.len();
    let (mut value, mut grad) = obj.value_and_gradient(pos.view());
    let mut h = Array2::<f64>::eye(n);
    let mut istep = control.istep;

    for step in 0..control.maxiter {
        let gnorm = l2(&grad);
        if gnorm < control.gtol {
            return Ok(done(value, pos, step, gnorm));
        }
        let dir = -h.dot(&grad);
        let old = pos.clone();
        let gold = grad.clone();
        let (npos, _, lsstep, moved) =
            take_step(obj, &pos, value, dir.view(), istep, linesearch, control);
        pos = npos;
        let ev = obj.value_and_gradient(pos.view());
        value = ev.0;
        grad = ev.1;
        if moved {
            let s = &pos - &old;
            let y = &grad - &gold;
            bfgs_inverse_update(&mut h, &s, &y);
        }
        istep = next_istep(lsstep, control);
    }
    Ok(done(value, pos, control.maxiter, l2(&grad)))
}

/// L-BFGS two-loop recursion (Nocedal-Wright 7.4).
pub fn minimize_lbfgs<O>(
    obj: &O,
    init: impl Into<Array1<f64>>,
    control: &Control,
    linesearch: LineSearch,
    memory: usize,
) -> Result<Report>
where
    O: DifferentiableObjective<f64> + ?Sized,
{
    let mut pos = start(obj, init)?;
    let (mut value, mut grad) = obj.value_and_gradient(pos.view());
    let mut s_hist: Vec<Array1<f64>> = Vec::new();
    let mut y_hist: Vec<Array1<f64>> = Vec::new();
    let mut istep = control.istep;
    let mcap = memory.max(1);

    for step in 0..control.maxiter {
        let gnorm = l2(&grad);
        if gnorm < control.gtol {
            return Ok(done(value, pos, step, gnorm));
        }
        let dir = lbfgs_direction(&grad, &s_hist, &y_hist);
        let old = pos.clone();
        let gold = grad.clone();
        let (npos, _, lsstep, moved) =
            take_step(obj, &pos, value, dir.view(), istep, linesearch, control);
        pos = npos;
        let ev = obj.value_and_gradient(pos.view());
        value = ev.0;
        grad = ev.1;
        if moved {
            let s = &pos - &old;
            let y = &grad - &gold;
            if y.dot(&s) > CURVATURE {
                if s_hist.len() == mcap {
                    s_hist.remove(0);
                    y_hist.remove(0);
                }
                s_hist.push(s);
                y_hist.push(y);
            }
        }
        istep = next_istep(lsstep, control);
    }
    Ok(done(value, pos, control.maxiter, l2(&grad)))
}

/// Inverse SR1 (Nocedal-Wright 6.24).
pub fn minimize_sr1<O>(
    obj: &O,
    init: impl Into<Array1<f64>>,
    control: &Control,
    linesearch: LineSearch,
) -> Result<Report>
where
    O: DifferentiableObjective<f64> + ?Sized,
{
    let mut pos = start(obj, init)?;
    let n = pos.len();
    let (mut value, mut grad) = obj.value_and_gradient(pos.view());
    let mut h = Array2::<f64>::eye(n);
    let mut istep = control.istep;

    for step in 0..control.maxiter {
        let gnorm = l2(&grad);
        if gnorm < control.gtol {
            return Ok(done(value, pos, step, gnorm));
        }
        let dir = -h.dot(&grad);
        let old = pos.clone();
        let gold = grad.clone();
        let (npos, _, lsstep, moved) =
            take_step(obj, &pos, value, dir.view(), istep, linesearch, control);
        pos = npos;
        let ev = obj.value_and_gradient(pos.view());
        value = ev.0;
        grad = ev.1;
        if moved {
            let s = &pos - &old;
            let y = &grad - &gold;
            sr1_inverse_update(&mut h, &s, &y);
        }
        istep = next_istep(lsstep, control);
    }
    Ok(done(value, pos, control.maxiter, l2(&grad)))
}

/// SR2 Hessian update from xtsci-optimize `sr2.hpp`.
///
/// Maintains a dense Hessian `B` (starts as `I`). The search direction is
/// `-B^{-1} g`. After a move, `B += δ(y + Bs)^T / (δ^T s)` with `δ = y - Bs`.
pub fn minimize_sr2<O>(
    obj: &O,
    init: impl Into<Array1<f64>>,
    control: &Control,
    linesearch: LineSearch,
) -> Result<Report>
where
    O: DifferentiableObjective<f64> + ?Sized,
{
    let mut pos = start(obj, init)?;
    let n = pos.len();
    let (mut value, mut grad) = obj.value_and_gradient(pos.view());
    let mut b = Array2::<f64>::eye(n);
    let mut istep = control.istep;

    for step in 0..control.maxiter {
        let gnorm = l2(&grad);
        if gnorm < control.gtol {
            return Ok(done(value, pos, step, gnorm));
        }
        let rhs = grad.mapv(|g| -g);
        let dir = solve_dense(&b, &rhs).unwrap_or_else(|| rhs);
        let old = pos.clone();
        let gold = grad.clone();
        let (npos, _, lsstep, moved) =
            take_step(obj, &pos, value, dir.view(), istep, linesearch, control);
        pos = npos;
        let ev = obj.value_and_gradient(pos.view());
        value = ev.0;
        grad = ev.1;
        if moved {
            let s = &pos - &old;
            let y = &grad - &gold;
            sr2_hessian_update(&mut b, &s, &y);
        }
        istep = next_istep(lsstep, control);
    }
    Ok(done(value, pos, control.maxiter, l2(&grad)))
}

/// Steepest descent: `d = -g` every iteration.
pub fn minimize_sd<O>(
    obj: &O,
    init: impl Into<Array1<f64>>,
    control: &Control,
    linesearch: LineSearch,
) -> Result<Report>
where
    O: DifferentiableObjective<f64> + ?Sized,
{
    let mut pos = start(obj, init)?;
    let (mut value, mut grad) = obj.value_and_gradient(pos.view());
    let mut istep = control.istep;
    for step in 0..control.maxiter {
        let gnorm = l2(&grad);
        if gnorm < control.gtol {
            return Ok(done(value, pos, step, gnorm));
        }
        let dir = grad.mapv(|g| -g);
        let (npos, _, lsstep, _) =
            take_step(obj, &pos, value, dir.view(), istep, linesearch, control);
        pos = npos;
        let ev = obj.value_and_gradient(pos.view());
        value = ev.0;
        grad = ev.1;
        istep = next_istep(lsstep, control);
    }
    Ok(done(value, pos, control.maxiter, l2(&grad)))
}

fn start<O>(obj: &O, init: impl Into<Array1<f64>>) -> Result<Array1<f64>>
where
    O: DifferentiableObjective<f64> + ?Sized,
{
    let pos = init.into();
    if pos.len() != Objective::dim(obj) {
        return Err(Error::Dim {
            got: pos.len(),
            dim: Objective::dim(obj),
        });
    }
    Ok(obj.bounds().clip(pos.view()))
}

fn done(value: f64, coords: Array1<f64>, steps: usize, grad_norm: f64) -> Report {
    Report {
        value,
        coords,
        steps,
        grad_norm,
    }
}

fn bfgs_inverse_update(h: &mut Array2<f64>, s: &Array1<f64>, y: &Array1<f64>) {
    let ys = y.dot(s);
    if ys <= CURVATURE {
        return;
    }
    let rho = 1.0 / ys;
    let hy = h.dot(y);
    let yhy = y.dot(&hy);
    let n = s.len();
    // H+ = (I - ρ s y^T) H (I - ρ y s^T) + ρ s s^T
    let mut hp = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            let mut acc = h[(i, j)];
            acc -= rho * s[i] * hy[j];
            acc -= rho * hy[i] * s[j];
            acc += rho * rho * yhy * s[i] * s[j];
            acc += rho * s[i] * s[j];
            hp[(i, j)] = acc;
        }
    }
    *h = hp;
}

fn sr2_hessian_update(b: &mut Array2<f64>, s: &Array1<f64>, y: &Array1<f64>) {
    let bs = b.dot(s);
    let delta = y - &bs;
    let ybs = y + &bs;
    let denom = delta.dot(s);
    let dn = l2(&delta);
    let sn = l2(s);
    if denom.abs() < SR1_SKIP * dn * sn || denom.abs() <= CURVATURE {
        return;
    }
    let n = s.len();
    for i in 0..n {
        for j in 0..n {
            b[(i, j)] += delta[i] * ybs[j] / denom;
        }
    }
}

/// Gaussian elimination with partial pivoting for `A x = rhs`.
fn solve_dense(a: &Array2<f64>, rhs: &Array1<f64>) -> Option<Array1<f64>> {
    let n = rhs.len();
    let mut m = a.clone();
    let mut x = rhs.clone();
    for k in 0..n {
        let mut piv = k;
        let mut best = m[(k, k)].abs();
        for i in (k + 1)..n {
            let v = m[(i, k)].abs();
            if v > best {
                best = v;
                piv = i;
            }
        }
        if best <= CURVATURE {
            return None;
        }
        if piv != k {
            for j in 0..n {
                let tmp = m[(k, j)];
                m[(k, j)] = m[(piv, j)];
                m[(piv, j)] = tmp;
            }
            let tmp = x[k];
            x[k] = x[piv];
            x[piv] = tmp;
        }
        let akk = m[(k, k)];
        for i in (k + 1)..n {
            let f = m[(i, k)] / akk;
            m[(i, k)] = 0.0;
            for j in (k + 1)..n {
                m[(i, j)] -= f * m[(k, j)];
            }
            x[i] -= f * x[k];
        }
    }
    for i in (0..n).rev() {
        let mut acc = x[i];
        for j in (i + 1)..n {
            acc -= m[(i, j)] * x[j];
        }
        let di = m[(i, i)];
        if di.abs() <= CURVATURE {
            return None;
        }
        x[i] = acc / di;
    }
    Some(x)
}

fn sr1_inverse_update(h: &mut Array2<f64>, s: &Array1<f64>, y: &Array1<f64>) {
    let hy = h.dot(y);
    let u = s - &hy;
    let uy = u.dot(y);
    let un = l2(&u);
    let yn = l2(y);
    if uy.abs() < SR1_SKIP * un * yn || uy.abs() <= CURVATURE {
        return;
    }
    let n = s.len();
    for i in 0..n {
        for j in 0..n {
            h[(i, j)] += u[i] * u[j] / uy;
        }
    }
}

fn lbfgs_direction(g: &Array1<f64>, s_hist: &[Array1<f64>], y_hist: &[Array1<f64>]) -> Array1<f64> {
    let m = s_hist.len();
    let mut q = g.clone();
    let mut alpha = vec![0.0; m];
    for i in (0..m).rev() {
        let ys = y_hist[i].dot(&s_hist[i]);
        if ys.abs() <= CURVATURE {
            continue;
        }
        let rho = 1.0 / ys;
        alpha[i] = rho * s_hist[i].dot(&q);
        for k in 0..q.len() {
            q[k] -= alpha[i] * y_hist[i][k];
        }
    }
    if m > 0 {
        let yy = y_hist[m - 1].dot(&y_hist[m - 1]);
        let sy = s_hist[m - 1].dot(&y_hist[m - 1]);
        if yy > CURVATURE {
            q.mapv_inplace(|qi| qi * (sy / yy));
        }
    }
    let mut r = q;
    for i in 0..m {
        let ys = y_hist[i].dot(&s_hist[i]);
        if ys.abs() <= CURVATURE {
            continue;
        }
        let rho = 1.0 / ys;
        let beta = rho * y_hist[i].dot(&r);
        for k in 0..r.len() {
            r[k] += (alpha[i] - beta) * s_hist[i][k];
        }
    }
    r.mapv(|ri| -ri)
}
