//! Stiefel \(\mathrm{St}(n,p)\): orthonormal n-by-p frames.
//!
//! `p = 1` is packed as a length-\(n\) unit vector and is the sphere.
//! `p > 1` is packed column-major (manopt `X(:)`), length `n*p`,
//! with \(X^\top X = I_p\). Projection is the horizontal space
//! \(U - X\,\mathrm{sym}(X^\top U)\). Retraction is thin QR with
//! positive diagonal (manopt `retr_qr` / `qr_unique`).
//!
//! A 3N cluster is [`super::RigidQuotient`], not \(\mathrm{St}(3N, 1)\).

use ndarray::{Array1, ArrayView1};

use crate::vecops::{dot, nrm2};

use super::{Manifold, sphere::Sphere};

/// Stiefel \(\mathrm{St}(n,p)\).
///
/// [`Default`] is `p = 1` with `n` taken from the vector (the
/// historical length-\(n\) packing). [`Stiefel::new`] fixes both
/// dimensions so a length-`n*p` token names `p`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stiefel {
    /// Rows. Zero means infer from the vector; only legal when `p = 1`.
    n: usize,
    /// Orthonormal columns.
    p: usize,
}

impl Default for Stiefel {
    fn default() -> Self {
        Self { n: 0, p: 1 }
    }
}

impl Stiefel {
    /// \(\mathrm{St}(n,p)\) with \(n \ge p \ge 1\).
    pub fn new(n: usize, p: usize) -> Result<Self, (usize, usize)> {
        if n >= p && p >= 1 {
            Ok(Self { n, p })
        } else {
            Err((n, p))
        }
    }

    /// Historical \(\mathrm{St}(*,1)\): length-\(n\) unit vector.
    pub fn p1() -> Self {
        Self { n: 0, p: 1 }
    }

    /// Ambient row count. Zero on the inferring `p = 1` token.
    pub fn rows(self) -> usize {
        self.n
    }

    /// Column count. `1` is the sphere packing.
    pub fn columns(self) -> usize {
        self.p
    }

    /// Packed length `n*p`, or `None` when `n` is inferred.
    pub fn packed_len(self) -> Option<usize> {
        if self.n == 0 {
            None
        } else {
            Some(self.n * self.p)
        }
    }

    /// Column-major n-by-p frame as a length-`n*p` token (manopt `X(:)`).
    pub fn pack(&self, frame: &[f64]) -> Option<Array1<f64>> {
        if !self.fits(frame.len()) {
            return None;
        }
        Some(Array1::from(frame.to_vec()))
    }

    /// Column-major storage of a packed point or tangent.
    pub fn unpack(&self, x: &Array1<f64>) -> Option<Vec<f64>> {
        if !self.fits(x.len()) {
            return None;
        }
        Some(x.to_vec())
    }

    fn fits(self, len: usize) -> bool {
        if self.p <= 1 {
            self.n == 0 || len == self.n
        } else {
            self.n > 0 && len == self.n * self.p
        }
    }

    fn from_len(len: usize, p: usize) -> Self {
        if p <= 1 {
            return Self::p1();
        }
        if p > 0 && len % p == 0 {
            let n = len / p;
            if n >= p {
                return Self { n, p };
            }
        }
        Self { n: 0, p }
    }

    fn at(n: usize, a: &[f64], i: usize, j: usize) -> f64 {
        a[i + n * j]
    }

    fn xtu(&self, x: &[f64], u: &[f64]) -> Vec<f64> {
        let mut s = vec![0.0; self.p * self.p];
        for a in 0..self.p {
            let xa = ArrayView1::from(&x[a * self.n..(a + 1) * self.n]);
            for b in 0..self.p {
                let ub = ArrayView1::from(&u[b * self.n..(b + 1) * self.n]);
                s[a + self.p * b] = dot(xa, ub);
            }
        }
        s
    }

    fn sym_inplace(s: &mut [f64], p: usize) {
        for a in 0..p {
            for b in 0..a {
                let v = 0.5 * (s[a + p * b] + s[b + p * a]);
                s[a + p * b] = v;
                s[b + p * a] = v;
            }
        }
    }

    fn qr_unique(&self, y: &mut [f64]) {
        for j in 0..self.p {
            for a in 0..j {
                let qa: Vec<f64> = y[a * self.n..(a + 1) * self.n].to_vec();
                let yj = ArrayView1::from(&y[j * self.n..(j + 1) * self.n]);
                let r = dot(ArrayView1::from(qa.as_slice()), yj);
                let yj = &mut y[j * self.n..(j + 1) * self.n];
                for (yi, qi) in yj.iter_mut().zip(&qa) {
                    *yi -= r * qi;
                }
            }
            let col = ArrayView1::from(&y[j * self.n..(j + 1) * self.n]);
            let nrm = nrm2(col);
            if nrm <= 1e-16 {
                continue;
            }
            for yi in &mut y[j * self.n..(j + 1) * self.n] {
                *yi /= nrm;
            }
        }
    }

    fn is_p1(self) -> bool {
        self.p <= 1
    }
}

impl Manifold for Stiefel {
    fn required_dim(&self, n: usize) -> Result<(), usize> {
        if self.fits(n) {
            Ok(())
        } else {
            Err(self.packed_len().unwrap_or(n))
        }
    }

    fn project(&self, x: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
        if self.is_p1() {
            if self.fits(x.len()) && x.len() == v.len() {
                return Sphere.project(x, v);
            }
            return v.clone();
        }
        let (Some(xv), Some(uv)) = (self.unpack(x), self.unpack(v)) else {
            return v.clone();
        };
        let mut xtu = self.xtu(&xv, &uv);
        Self::sym_inplace(&mut xtu, self.p);
        let mut out = uv;
        for j in 0..self.p {
            for i in 0..self.n {
                let mut acc = 0.0;
                for a in 0..self.p {
                    acc += Self::at(self.n, &xv, i, a) * xtu[a + self.p * j];
                }
                out[i + self.n * j] -= acc;
            }
        }
        Array1::from(out)
    }

    fn retract(&self, x: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
        if self.is_p1() {
            if self.fits(x.len()) && x.len() == v.len() {
                return Sphere.retract(x, v);
            }
            return x + v;
        }
        let (Some(xv), Some(uv)) = (self.unpack(x), self.unpack(v)) else {
            return x + v;
        };
        let mut y = xv;
        for (yi, ui) in y.iter_mut().zip(&uv) {
            *yi += *ui;
        }
        self.qr_unique(&mut y);
        Array1::from(y)
    }

    fn transport(&self, x_from: &Array1<f64>, x_to: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
        if self.is_p1() {
            return Sphere.transport(x_from, x_to, v);
        }
        self.project(x_to, v)
    }
}

/// Build \(\mathrm{St}(n,p)\) from a packed length and a column count.
pub(crate) fn stiefel_from_len(len: usize, p: usize) -> Stiefel {
    Stiefel::from_len(len, p)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn frame_4x2() -> Array1<f64> {
        // Columns (1,0,0,0) and (0,1,0,0), column-major.
        array![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    }

    #[test]
    fn p1_matches_sphere_in_all_three_operations() {
        let st = Stiefel::p1();
        let x = array![0.6, 0.8, 0.0];
        let v = array![0.1, -0.2, 0.5];
        let y = array![0.0, 1.0, 0.0];
        assert_eq!(st.project(&x, &v), Sphere.project(&x, &v));
        assert_eq!(st.retract(&x, &v), Sphere.retract(&x, &v));
        assert_eq!(st.transport(&x, &y, &v), Sphere.transport(&x, &y, &v));
    }

    #[test]
    fn project_is_horizontal() {
        let st = Stiefel::new(4, 2).unwrap();
        let x = frame_4x2();
        let v = array![0.2, 0.1, 0.3, -0.4, 0.5, -0.2, 0.1, 0.7];
        let z = st.project(&x, &v);
        let xtz = st.xtu(x.as_slice().unwrap(), z.as_slice().unwrap());
        for a in 0..2 {
            for b in 0..2 {
                let s = xtz[a + 2 * b] + xtz[b + 2 * a];
                assert!(s.abs() < 1e-12, "X^T Z + Z^T X [{a},{b}] = {s}");
            }
        }
    }

    #[test]
    fn retract_stays_on_stiefel() {
        let st = Stiefel::new(4, 2).unwrap();
        let x = frame_4x2();
        let v = st.project(&x, &array![0.1, 0.0, 0.2, 0.0, 0.0, 0.1, 0.0, -0.2]);
        let y = st.retract(&x, &v);
        let yty = st.xtu(y.as_slice().unwrap(), y.as_slice().unwrap());
        for a in 0..2 {
            for b in 0..2 {
                let got = yty[a + 2 * b];
                let want = if a == b { 1.0 } else { 0.0 };
                assert!((got - want).abs() < 1e-12, "Y^T Y[{a},{b}]={got}");
            }
        }
    }

    #[test]
    fn pack_unpack_is_column_major() {
        let st = Stiefel::new(3, 2).unwrap();
        let frame = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let x = st.pack(&frame).unwrap();
        assert_eq!(x.len(), 6);
        assert_eq!(st.unpack(&x).unwrap(), frame);
        assert!(st.pack(&[1.0, 0.0, 0.0]).is_none());
    }

    #[test]
    fn p_greater_than_one_rejects_a_3n_cluster_length() {
        let st = Stiefel::new(4, 2).unwrap();
        assert!(st.required_dim(8).is_ok());
        assert!(st.required_dim(12).is_err());
        assert!(st.required_dim(114).is_err());
        assert!(Stiefel::p1().required_dim(114).is_ok());
        assert!(Stiefel::new(3, 4).is_err());
    }

    #[test]
    fn wrong_dim_keeps_length() {
        let st = Stiefel::new(4, 2).unwrap();
        let x = Array1::from_elem(12, 0.1);
        let v = Array1::from_elem(12, 0.01);
        let y = st.retract(&x, &v);
        assert_eq!(y.len(), 12);
        assert_eq!(st.project(&x, &v).len(), 12);
    }
}
