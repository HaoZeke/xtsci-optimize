//! Matrix-free lowest eigenpair of a symmetric Hessian action.
//!
//! IRC kick and the `lambda_min` sign check need one extremal pair,
//! not a full ELPA / SLATE spectrum. This is Lanczos on
//! [`HessianVector`] with a tiny Jacobi solve on the tridiagonal,
//! the same algorithm rgsaddle uses for min-mode.

use ndarray::{Array1, ArrayView1};

use crate::hvp::HessianVector;
use crate::vecops::{axpy, dot, nrm2};

/// Result of [`lowest_eigenpair`].
#[derive(Clone, Debug)]
pub struct LowestMode {
    /// Approximate eigenvector, Euclidean-normalized.
    pub vector: Array1<f64>,
    /// Approximate eigenvalue (Rayleigh quotient).
    pub value: f64,
    /// Number of Hessian actions.
    pub actions: usize,
}

/// Lanczos for the lowest eigenpair of `H(x)`.
pub fn lowest_eigenpair<H: HessianVector + ?Sized>(
    h: &H,
    x: ArrayView1<f64>,
    seed: ArrayView1<f64>,
    krylov: usize,
) -> LowestMode {
    let n = seed.len();
    let m = krylov.min(n).max(1);
    let mut q: Vec<Array1<f64>> = Vec::with_capacity(m);
    let mut alpha = Vec::with_capacity(m);
    let mut beta: Vec<f64> = Vec::with_capacity(m);
    q.push(normalize(seed.to_owned()));

    let mut actions = 0;
    for j in 0..m {
        let hv = h.hessian_vector(x, q[j].view());
        actions += 1;
        let a = dot(hv.view(), q[j].view());
        alpha.push(a);
        if j + 1 == m {
            break;
        }
        let mut w = hv;
        axpy(-a, q[j].view(), &mut w);
        if j > 0 {
            axpy(-beta[j - 1], q[j - 1].view(), &mut w);
        }
        for qi in q.iter() {
            let overlap = dot(w.view(), qi.view());
            axpy(-overlap, qi.view(), &mut w);
        }
        let b = nrm2(w.view());
        if b <= 1e-12 {
            break;
        }
        beta.push(b);
        q.push(w / b);
    }

    let k = alpha.len();
    let mut t = vec![vec![0.0; k]; k];
    for i in 0..k {
        t[i][i] = alpha[i];
        if i + 1 < k && i < beta.len() {
            t[i][i + 1] = beta[i];
            t[i + 1][i] = beta[i];
        }
    }
    let (evals, evecs) = jacobi_eigen(&mut t);
    let mut lowest = 0;
    for i in 1..k {
        if evals[i] < evals[lowest] {
            lowest = i;
        }
    }
    let mut mode = Array1::zeros(n);
    for (i, qi) in q.iter().enumerate().take(k) {
        axpy(evecs[i][lowest], qi.view(), &mut mode);
    }
    LowestMode {
        vector: normalize(mode),
        value: evals[lowest],
        actions,
    }
}

fn normalize(mut v: Array1<f64>) -> Array1<f64> {
    let n = nrm2(v.view());
    if n > 1e-14 {
        v.mapv_inplace(|c| c / n);
    }
    v
}

fn jacobi_eigen(a: &mut [Vec<f64>]) -> (Vec<f64>, Vec<Vec<f64>>) {
    let n = a.len();
    let mut v = vec![vec![0.0; n]; n];
    for (i, row) in v.iter_mut().enumerate() {
        row[i] = 1.0;
    }
    for _ in 0..64 {
        let mut off = 0.0;
        for i in 0..n {
            for j in (i + 1)..n {
                let aij = a[i][j];
                off += aij * aij;
                if aij.abs() <= 1e-15 {
                    continue;
                }
                let tau = (a[j][j] - a[i][i]) / (2.0 * aij);
                let t = if tau >= 0.0 {
                    1.0 / (tau + (1.0 + tau * tau).sqrt())
                } else {
                    -1.0 / (-tau + (1.0 + tau * tau).sqrt())
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;
                let aii = a[i][i];
                let ajj = a[j][j];
                a[i][i] = c * c * aii - 2.0 * s * c * aij + s * s * ajj;
                a[j][j] = s * s * aii + 2.0 * s * c * aij + c * c * ajj;
                a[i][j] = 0.0;
                a[j][i] = 0.0;
                for k in 0..n {
                    if k != i && k != j {
                        let aik = a[i][k];
                        let ajk = a[j][k];
                        a[i][k] = c * aik - s * ajk;
                        a[k][i] = a[i][k];
                        a[j][k] = s * aik + c * ajk;
                        a[k][j] = a[j][k];
                    }
                    let vki = v[k][i];
                    let vkj = v[k][j];
                    v[k][i] = c * vki - s * vkj;
                    v[k][j] = s * vki + c * vkj;
                }
            }
        }
        if off.sqrt() <= 1e-14 {
            break;
        }
    }
    let evals: Vec<f64> = (0..n).map(|i| a[i][i]).collect();
    (evals, v)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hvp::HvpOracle;
    use ndarray::array;

    #[test]
    fn recovers_the_downhill_axis_of_a_double_well_hessian() {
        // H = diag(-8, 4, 4, 4, 4, 4) at the origin of (x^2-1)^2 + 2 r_rest^2.
        let h = HvpOracle::unbounded(
            6,
            |x| {
                let t = x[0];
                let mut g = Array1::zeros(6);
                g[0] = 4.0 * t * (t * t - 1.0);
                for i in 1..6 {
                    g[i] = 4.0 * x[i];
                }
                let e = (t * t - 1.0) * (t * t - 1.0)
                    + 2.0 * x.iter().skip(1).map(|c| c * c).sum::<f64>();
                (e, g)
            },
            |_x, v| {
                let mut hv = Array1::zeros(6);
                hv[0] = -8.0 * v[0];
                for i in 1..6 {
                    hv[i] = 4.0 * v[i];
                }
                hv
            },
        );
        let x = Array1::zeros(6);
        let seed = array![0.2, 0.7, 0.1, 0.0, 0.0, 0.0];
        let mode = lowest_eigenpair(&h, x.view(), seed.view(), 6);
        assert!(mode.value < 0.0, "saddle curvature {}", mode.value);
        assert!(
            mode.vector[0].abs() > 0.9,
            "mode should lie on x, got {:?}",
            mode.vector
        );
        assert!(mode.actions <= 6);
    }
}
