//! The vector seam: every O(n) operation a solver performs, in one
//! place.
//!
//! TAO writes each algorithm against PETSc `Vec` -- `VecDot`,
//! `VecAXPY`, `VecNorm` -- and that seam is its entire scaling story:
//! one `blmvm.c` runs serial, distributed, or on device because the
//! solver never touches storage. Here the objective side already has
//! its seam (`eindir` supplies `f` and `g`); this module is the
//! matching one for the solver's own algebra. Algorithms call these
//! functions; storage-specific execution lives behind them.
//!
//! The default implementation is the serial one the whole test suite
//! pins bit-identically. The `par` feature swaps in rayon over slices
//! for the length-`n` reductions and updates; results may differ from
//! serial in the last bits because floating-point addition reorders,
//! which is why it is a feature and not a default, and why no test of
//! the serial path runs under it.

use ndarray::{Array1, ArrayView1};

/// `x . y`.
#[cfg(not(feature = "par"))]
pub fn dot(x: ArrayView1<f64>, y: ArrayView1<f64>) -> f64 {
    x.dot(&y)
}

/// `x . y`, reduced in parallel chunks.
#[cfg(feature = "par")]
pub fn dot(x: ArrayView1<f64>, y: ArrayView1<f64>) -> f64 {
    use rayon::prelude::*;
    match (x.as_slice(), y.as_slice()) {
        (Some(a), Some(b)) => a
            .par_chunks(4096)
            .zip(b.par_chunks(4096))
            .map(|(ca, cb)| ca.iter().zip(cb).map(|(p, q)| p * q).sum::<f64>())
            .sum(),
        _ => x.dot(&y),
    }
}

/// `y += a x`.
#[cfg(not(feature = "par"))]
pub fn axpy(a: f64, x: ArrayView1<f64>, y: &mut Array1<f64>) {
    y.scaled_add(a, &x);
}

/// `y += a x`, updated in parallel chunks.
#[cfg(feature = "par")]
pub fn axpy(a: f64, x: ArrayView1<f64>, y: &mut Array1<f64>) {
    use rayon::prelude::*;
    match (x.as_slice(), y.as_slice_mut()) {
        (Some(xs), Some(ys)) => ys
            .par_chunks_mut(4096)
            .zip(xs.par_chunks(4096))
            .for_each(|(cy, cx)| {
                for (p, q) in cy.iter_mut().zip(cx) {
                    *p += a * q;
                }
            }),
        _ => y.scaled_add(a, &x),
    }
}

/// `||x||_2`.
pub fn nrm2(x: ArrayView1<f64>) -> f64 {
    dot(x, x).sqrt()
}

/// `||x||_inf`, infinity on any non-finite component: a broken vector
/// is never a small one.
pub fn nrminf(x: ArrayView1<f64>) -> f64 {
    let mut m = 0.0_f64;
    for v in x.iter() {
        if !v.is_finite() {
            return f64::INFINITY;
        }
        m = m.max(v.abs());
    }
    m
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    #[test]
    fn the_seam_matches_ndarray_on_the_serial_path() {
        let x = Array1::from((0..1000).map(|i| (i as f64).sin()).collect::<Vec<_>>());
        let y = Array1::from((0..1000).map(|i| (i as f64).cos()).collect::<Vec<_>>());
        let d = dot(x.view(), y.view());
        let mut z = y.clone();
        axpy(0.5, x.view(), &mut z);
        let mut reference = y.clone();
        reference.scaled_add(0.5, &x);
        assert!((d - x.dot(&y)).abs() < 1e-12 * d.abs().max(1.0));
        assert_eq!(z, reference);
        assert!((nrm2(x.view()) - x.dot(&x).sqrt()).abs() < 1e-12);
    }

    #[test]
    fn the_infinity_norm_never_calls_a_broken_vector_small() {
        let x = Array1::from(vec![0.0, f64::NAN, 1.0]);
        assert!(nrminf(x.view()).is_infinite());
    }
}
