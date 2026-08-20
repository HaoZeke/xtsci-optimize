//! Bound-constrained L-BFGS quadratic model, solved by HiGHS.
//!
//! The compact Hessian (Byrd, Nocedal, Schnabel; Nocedal-Wright 7.19)
//! is `B = θ I - W M^{-1} W^T`. Each step solves
//! `min g·p + (1/2) p^T B p` subject to an L_inf trust region, optional
//! box bounds, and optional linear equalities.
//!
//! Unconstrained, this QP is the two-loop direction. With a box it is
//! the L-BFGS-B model without the Cauchy / subspace split: HiGHS takes
//! the whole convex QP.
//!
//! Huangfu and Hall, *Parallelizing the dual revised simplex method*,
//! <https://doi.org/10.1007/s12532-017-0130-5>.

use highs::{HighsModelStatus, RowProblem, Sense};
use highs_sys::{Highs_passHessian, HighsInt, STATUS_OK};
use ndarray::{Array1, Array2, ArrayView1};

use crate::error::{Error, Result};
use crate::lbfgs::Lbfgs;

const DIAG_FLOOR: f64 = 1e-6;
const HESS_NZ: f64 = 1e-16;

/// Bounds and equalities on one L-BFGS model step.
#[derive(Clone, Debug, Default)]
pub struct HighsStep {
    /// L_inf trust radius on the step. `None` is unbounded.
    pub trust: Option<f64>,
    /// Uniform box lower bound on coordinates of `x + p`.
    pub lo: Option<f64>,
    /// Uniform box upper bound on coordinates of `x + p`.
    pub hi: Option<f64>,
    /// Linear equalities `a · p = rhs`.
    pub equalities: Vec<(Vec<(usize, f64)>, f64)>,
}

impl Lbfgs {
    /// Solves the L-BFGS quadratic model at `x` with gradient `g`.
    pub fn highs_step(&self, x: ArrayView1<f64>, g: ArrayView1<f64>) -> Result<Array1<f64>> {
        let opts = self.highs.as_ref().ok_or_else(|| {
            Error::Highs("Lbfgs.highs is None; set HighsStep before highs_step".into())
        })?;
        if x.len() != g.len() {
            return Err(Error::Dim {
                got: x.len(),
                dim: g.len(),
            });
        }
        let n = x.len();
        let (theta, b) = self.compact_hessian(n)?;
        let _ = theta;
        match qp_step(&b, x, g, opts) {
            Ok(p) => Ok(p),
            Err(_) => Ok(clip_step(self.direction(g), x, opts)),
        }
    }

    /// Compact L-BFGS Hessian `B` (Nocedal-Wright 7.19) and scale `θ`.
    pub(crate) fn compact_hessian(&self, n: usize) -> Result<(f64, Array2<f64>)> {
        let m = self.memory_len();
        if m == 0 {
            return Ok((1.0, Array2::<f64>::eye(n)));
        }
        let last = self.pair(m - 1);
        let yy = last.1.dot(last.1);
        let sy = last.0.dot(last.1);
        let theta = if sy > HESS_NZ { yy / sy } else { 1.0 };

        let mut w = Array2::<f64>::zeros((n, 2 * m));
        for j in 0..m {
            let (s, y) = self.pair(j);
            for i in 0..n {
                w[(i, j)] = theta * s[i];
                w[(i, m + j)] = y[i];
            }
        }

        let mut mat = Array2::<f64>::zeros((2 * m, 2 * m));
        for i in 0..m {
            let (si, _) = self.pair(i);
            for j in 0..m {
                let (sj, _) = self.pair(j);
                mat[(i, j)] = theta * si.dot(sj);
            }
        }
        for i in 0..m {
            let (si, _) = self.pair(i);
            for j in 0..m {
                if i > j {
                    let (_, yj) = self.pair(j);
                    let lij = si.dot(yj);
                    mat[(i, m + j)] = lij;
                    mat[(m + j, i)] = lij;
                }
            }
        }
        for i in 0..m {
            let (si, yi) = self.pair(i);
            mat[(m + i, m + i)] = -si.dot(yi);
        }

        let minv = invert_dense(&mat)
            .ok_or_else(|| Error::Highs("compact L-BFGS M is singular".into()))?;
        let mid = minv.dot(&w.t());
        let lowrank = w.dot(&mid);
        let mut b = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                let mut v = -lowrank[(i, j)];
                if i == j {
                    v += theta;
                }
                b[(i, j)] = v;
            }
        }
        Ok((theta, b))
    }
}

fn qp_step(
    b: &Array2<f64>,
    x: ArrayView1<f64>,
    g: ArrayView1<f64>,
    opts: &HighsStep,
) -> Result<Array1<f64>> {
    let n = x.len();
    let mut pb = RowProblem::default();
    let mut cols = Vec::with_capacity(n);
    for k in 0..n {
        let mut lo = opts.trust.map(|t| -t).unwrap_or(f64::NEG_INFINITY);
        let mut hi = opts.trust.map(|t| t).unwrap_or(f64::INFINITY);
        if let Some(b0) = opts.lo {
            lo = lo.max(b0 - x[k]);
        }
        if let Some(b1) = opts.hi {
            hi = hi.min(b1 - x[k]);
        }
        if lo > hi {
            lo = hi;
        }
        cols.push(pb.add_column(g[k], lo..=hi));
    }
    for (coeffs, rhs) in &opts.equalities {
        let row: Vec<_> = coeffs.iter().map(|(i, a)| (cols[*i], *a)).collect();
        pb.add_row(*rhs..=*rhs, &row);
    }

    let mut model = pb
        .try_optimise(Sense::Minimise)
        .map_err(|e| Error::Highs(format!("pass LP {e:?}")))?;
    model.make_quiet();
    // HiGHS default OpenMP workers deadlock Rayon in the next eval
    // (landfold stress par_iter). Pin OpenMP before Highs_run so the
    // runtime never starts a second pool.
    unsafe {
        std::env::set_var("OMP_NUM_THREADS", "1");
    }
    model
        .try_set_option("parallel", "off")
        .map_err(|_| Error::Highs("cannot set parallel=off".into()))?;
    model
        .try_set_option("threads", 1_i32)
        .map_err(|_| Error::Highs("cannot set threads=1".into()))?;
    model
        .try_set_option("time_limit", 0.05_f64)
        .map_err(|_| Error::Highs("cannot set time_limit".into()))?;

    let (q_start, q_index, q_value) = lower_csc(b);
    let q_nnz = q_value.len();
    let st = unsafe {
        Highs_passHessian(
            model.as_mut_ptr(),
            n as HighsInt,
            q_nnz as HighsInt,
            1,
            q_start.as_ptr(),
            q_index.as_ptr(),
            q_value.as_ptr(),
        )
    };
    if st != STATUS_OK {
        return Err(Error::Highs(format!("pass Hessian status {st}")));
    }

    let solved = model
        .try_solve()
        .map_err(|e| Error::Highs(format!("solve {e:?}")))?;
    if solved.status() != HighsModelStatus::Optimal {
        return Err(Error::Highs(format!("status {:?}", solved.status())));
    }
    let sol = solved.get_solution();
    let p = sol.columns();
    if p.len() != n {
        return Err(Error::Highs(format!("column count {} != {n}", p.len())));
    }
    Ok(Array1::from(p.to_vec()))
}

fn clip_step(mut d: Array1<f64>, x: ArrayView1<f64>, opts: &HighsStep) -> Array1<f64> {
    for k in 0..d.len() {
        let mut lo = opts.trust.map(|t| -t).unwrap_or(f64::NEG_INFINITY);
        let mut hi = opts.trust.map(|t| t).unwrap_or(f64::INFINITY);
        if let Some(b0) = opts.lo {
            lo = lo.max(b0 - x[k]);
        }
        if let Some(b1) = opts.hi {
            hi = hi.min(b1 - x[k]);
        }
        if lo > hi {
            lo = hi;
        }
        d[k] = d[k].clamp(lo, hi);
    }
    d
}

fn lower_csc(b: &Array2<f64>) -> (Vec<HighsInt>, Vec<HighsInt>, Vec<f64>) {
    let n = b.nrows();
    let mut start = Vec::with_capacity(n);
    let mut index = Vec::new();
    let mut value = Vec::new();
    for j in 0..n {
        start.push(value.len() as HighsInt);
        for i in j..n {
            let mut v = b[(i, j)];
            if i == j {
                v = v.max(DIAG_FLOOR);
            }
            if i == j || v.abs() > HESS_NZ {
                index.push(i as HighsInt);
                value.push(v);
            }
        }
    }
    (start, index, value)
}

fn invert_dense(a: &Array2<f64>) -> Option<Array2<f64>> {
    let n = a.nrows();
    if n == 0 {
        return Some(Array2::zeros((0, 0)));
    }
    let mut m = a.clone();
    let mut inv = Array2::<f64>::eye(n);
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
        if best <= HESS_NZ {
            return None;
        }
        if piv != k {
            for j in 0..n {
                let tmp = m[(k, j)];
                m[(k, j)] = m[(piv, j)];
                m[(piv, j)] = tmp;
                let tmp = inv[(k, j)];
                inv[(k, j)] = inv[(piv, j)];
                inv[(piv, j)] = tmp;
            }
        }
        let akk = m[(k, k)];
        for j in 0..n {
            m[(k, j)] /= akk;
            inv[(k, j)] /= akk;
        }
        for i in 0..n {
            if i == k {
                continue;
            }
            let f = m[(i, k)];
            for j in 0..n {
                m[(i, j)] -= f * m[(k, j)];
                inv[(i, j)] -= f * inv[(k, j)];
            }
        }
    }
    Some(inv)
}
