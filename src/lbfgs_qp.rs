//! L-BFGS direction, HiGHS only for the feasible set.
//!
//! The two-loop recursion is the unconstrained minimizer of the L-BFGS
//! quadratic model (Nocedal-Wright 7.4). Forming the dense compact
//! Hessian and handing it to HiGHS is slower and, on a tiny accepted
//! step, indefinite: `Highs_run` does not return.
//!
//! Constrained step: project that direction onto the L_inf trust
//! region, the coordinate box, and any linear equalities. Box and
//! trust are coordinate clamps. Equalities go to HiGHS as
//! `min 1/2 ||p - d||^2` with `Q = I`, which is convex.
//!
//! Huangfu and Hall, *Parallelizing the dual revised simplex method*,
//! <https://doi.org/10.1007/s12532-017-0130-5>.

use highs::{HighsModelStatus, RowProblem, Sense};
use highs_sys::{Highs_passHessian, HighsInt, STATUS_OK};
use ndarray::{Array1, ArrayView1};

use crate::error::{Error, Result};
use crate::lbfgs::Lbfgs;

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
    /// Packed `(n_atoms, dim)`: enforce `sum_i p[i * dim + h] = 0` per axis.
    /// This is a mean subtract, not a QP.
    pub center_axes: Option<(usize, usize)>,
}

impl HighsStep {
    fn has_box(&self) -> bool {
        self.trust.is_some() || self.lo.is_some() || self.hi.is_some()
    }

    fn needs_qp(&self) -> bool {
        !self.equalities.is_empty()
    }
}

impl Lbfgs {
    /// L-BFGS direction at `x` with gradient `g`, projected if needed.
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
        let d = self.direction(g);
        if !opts.has_box() && !opts.needs_qp() && opts.center_axes.is_none() {
            return Ok(d);
        }
        let mut p = d;
        if let Some((n_atoms, dim)) = opts.center_axes {
            project_center_box(&mut p, x, opts, n_atoms, dim);
        } else {
            p = clip_step(p, x, opts);
        }
        if !opts.needs_qp() {
            return Ok(p);
        }
        match project_qp(&p, x, opts) {
            Ok(q) => Ok(q),
            Err(_) => Ok(p),
        }
    }
}

fn project_qp(d: &Array1<f64>, x: ArrayView1<f64>, opts: &HighsStep) -> Result<Array1<f64>> {
    let n = d.len();
    let mut pb = RowProblem::default();
    let mut cols = Vec::with_capacity(n);
    for k in 0..n {
        let (lo, hi) = column_bounds(k, x, opts);
        // min 1/2 ||p - d||^2  <=>  min -d · p + 1/2 p^T p
        cols.push(pb.add_column(-d[k], lo..=hi));
    }
    for (coeffs, rhs) in &opts.equalities {
        let row: Vec<_> = coeffs.iter().map(|(i, a)| (cols[*i], *a)).collect();
        pb.add_row(*rhs..=*rhs, &row);
    }

    let mut model = pb
        .try_optimise(Sense::Minimise)
        .map_err(|e| Error::Highs(format!("pass LP {e:?}")))?;
    model.make_quiet();
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

    let (q_start, q_index, q_value) = identity_csc(n);
    let st = unsafe {
        Highs_passHessian(
            model.as_mut_ptr(),
            n as HighsInt,
            n as HighsInt,
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

fn column_bounds(k: usize, x: ArrayView1<f64>, opts: &HighsStep) -> (f64, f64) {
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
    (lo, hi)
}

/// Dykstra on the intersection of the centering subspace and the box.
fn project_center_box(
    d: &mut Array1<f64>,
    x: ArrayView1<f64>,
    opts: &HighsStep,
    n_atoms: usize,
    dim: usize,
) {
    if n_atoms == 0 || dim == 0 || d.len() != n_atoms * dim {
        clip_in_place(d, x, opts);
        return;
    }
    for _ in 0..16 {
        center_axes(d, n_atoms, dim);
        clip_in_place(d, x, opts);
    }
    center_axes(d, n_atoms, dim);
}

fn center_axes(d: &mut Array1<f64>, n_atoms: usize, dim: usize) {
    let n = n_atoms as f64;
    for h in 0..dim {
        let mut sum = 0.0;
        for i in 0..n_atoms {
            sum += d[i * dim + h];
        }
        let mean = sum / n;
        for i in 0..n_atoms {
            d[i * dim + h] -= mean;
        }
    }
}

fn clip_in_place(d: &mut Array1<f64>, x: ArrayView1<f64>, opts: &HighsStep) {
    for k in 0..d.len() {
        let (lo, hi) = column_bounds(k, x, opts);
        d[k] = d[k].clamp(lo, hi);
    }
}

fn clip_step(mut d: Array1<f64>, x: ArrayView1<f64>, opts: &HighsStep) -> Array1<f64> {
    clip_in_place(&mut d, x, opts);
    d
}

fn identity_csc(n: usize) -> (Vec<HighsInt>, Vec<HighsInt>, Vec<f64>) {
    let start: Vec<HighsInt> = (0..n).map(|j| j as HighsInt).collect();
    let index: Vec<HighsInt> = (0..n).map(|j| j as HighsInt).collect();
    let value = vec![1.0; n];
    (start, index, value)
}
