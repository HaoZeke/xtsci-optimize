//! Real Grassmannian \(\mathrm{Gr}(n,p)\). manopt `grassmannfactory`.
//!
//! A point is an n-by-p orthonormal frame packed column-major
//! (length `n*p`). The geometry is the usual quotient of Stiefel:
//! only the column space matters. Projection is `U - X (X^T U)`.
//! Retraction is thin QR of `X+U` (manopt's cheaper alternative to
//! the polar factor; the column space is the same).
//!
//! `p == 1` is \(\mathrm{RP}^{n-1}\). A 3N cluster is
//! [`super::RigidQuotient`], not \(\mathrm{Gr}(3N, 1)\).

use ndarray::Array1;

use super::Manifold;

/// Real Grassmann \(\mathrm{Gr}(n,p)\). `n > p >= 1`.
#[derive(Clone, Copy, Debug)]
pub struct Grassmann {
    /// Ambient dimension.
    pub n: usize,
    /// Subspace dimension.
    pub p: usize,
}

impl Grassmann {
    /// `Ok` when `n > p` and `p >= 1`.
    pub fn new(n: usize, p: usize) -> Result<Self, (usize, usize)> {
        if n > p && p >= 1 {
            Ok(Self { n, p })
        } else {
            Err((n, p))
        }
    }

    fn unpack(&self, x: &Array1<f64>) -> Option<Vec<f64>> {
        if x.len() != self.n * self.p {
            return None;
        }
        Some(x.to_vec())
    }

    fn at(&self, a: &[f64], i: usize, j: usize) -> f64 {
        a[i + self.n * j]
    }

    fn set(&self, a: &mut [f64], i: usize, j: usize, v: f64) {
        a[i + self.n * j] = v;
    }

    fn xtu(&self, x: &[f64], u: &[f64]) -> Vec<f64> {
        let mut s = vec![0.0; self.p * self.p];
        for a in 0..self.p {
            for b in 0..self.p {
                let mut acc = 0.0;
                for i in 0..self.n {
                    acc += self.at(x, i, a) * self.at(u, i, b);
                }
                s[a + self.p * b] = acc;
            }
        }
        s
    }
}

impl Default for Grassmann {
    fn default() -> Self {
        Self { n: 2, p: 1 }
    }
}

impl Manifold for Grassmann {
    fn required_dim(&self, n: usize) -> Result<(), usize> {
        if n == self.n * self.p {
            Ok(())
        } else {
            Err(n)
        }
    }

    fn project(&self, x: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
        let (Some(xv), Some(uv)) = (self.unpack(x), self.unpack(v)) else {
            return v.clone();
        };
        let xtu = self.xtu(&xv, &uv);
        let mut out = uv;
        for j in 0..self.p {
            for i in 0..self.n {
                let mut acc = 0.0;
                for a in 0..self.p {
                    acc += self.at(&xv, i, a) * xtu[a + self.p * j];
                }
                let idx = i + self.n * j;
                out[idx] -= acc;
            }
        }
        Array1::from(out)
    }

    fn retract(&self, x: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
        let (Some(xv), Some(uv)) = (self.unpack(x), self.unpack(v)) else {
            return x.clone();
        };
        let mut y = xv;
        for i in 0..y.len() {
            y[i] += uv[i];
        }
        // Modified Gram-Schmidt on the p columns.
        for j in 0..self.p {
            for a in 0..j {
                let mut dot = 0.0;
                for i in 0..self.n {
                    dot += self.at(&y, i, a) * self.at(&y, i, j);
                }
                for i in 0..self.n {
                    let val = self.at(&y, i, j) - dot * self.at(&y, i, a);
                    self.set(&mut y, i, j, val);
                }
            }
            let mut nrm = 0.0;
            for i in 0..self.n {
                let t = self.at(&y, i, j);
                nrm += t * t;
            }
            nrm = nrm.sqrt();
            if nrm <= 1e-16 {
                continue;
            }
            for i in 0..self.n {
                let val = self.at(&y, i, j) / nrm;
                self.set(&mut y, i, j, val);
            }
        }
        Array1::from(y)
    }

    fn transport(&self, _x_from: &Array1<f64>, x_to: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
        self.project(x_to, v)
    }
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
    fn project_is_horizontal() {
        let g = Grassmann { n: 4, p: 2 };
        let x = frame_4x2();
        let v = array![0.2, 0.1, 0.3, -0.4, 0.5, -0.2, 0.1, 0.7];
        let t = g.project(&x, &v);
        // X^T t == 0
        let xtt = g.xtu(x.as_slice().unwrap(), t.as_slice().unwrap());
        for s in xtt {
            assert!(s.abs() < 1e-12, "{s}");
        }
    }

    #[test]
    fn retract_stays_on_stiefel() {
        let g = Grassmann { n: 4, p: 2 };
        let x = frame_4x2();
        let v = g.project(&x, &array![0.1, 0.0, 0.2, 0.0, 0.0, 0.1, 0.0, -0.2]);
        let y = g.retract(&x, &v);
        let yty = g.xtu(y.as_slice().unwrap(), y.as_slice().unwrap());
        for a in 0..2 {
            for b in 0..2 {
                let got = yty[a + 2 * b];
                let want = if a == b { 1.0 } else { 0.0 };
                assert!((got - want).abs() < 1e-12, "Y^T Y[{a},{b}]={got}");
            }
        }
    }
}
