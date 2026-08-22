//! SE(3): row-major SO(3) (9) then translation (3). Length 12.

use ndarray::Array1;

use super::{Manifold, so3::So3};

/// Rigid motions. Rotation block uses [`So3`]; translation is Euclidean.
#[derive(Clone, Copy, Debug, Default)]
pub struct Se3;

impl Manifold for Se3 {
    fn project(&self, x: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
        if x.len() < 12 || v.len() < 12 {
            return v.clone();
        }
        let xr = x.slice(ndarray::s![0..9]).to_owned();
        let vr = v.slice(ndarray::s![0..9]).to_owned();
        let pr = So3.project(&xr, &vr);
        let mut out = v.clone();
        for i in 0..9 {
            out[i] = pr[i];
        }
        out
    }

    fn retract(&self, x: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
        if x.len() < 12 || v.len() < 12 {
            return x + v;
        }
        let xr = x.slice(ndarray::s![0..9]).to_owned();
        let vr = v.slice(ndarray::s![0..9]).to_owned();
        let yr = So3.retract(&xr, &vr);
        let mut y = x + v;
        for i in 0..9 {
            y[i] = yr[i];
        }
        y
    }

    fn transport(&self, _x_from: &Array1<f64>, x_to: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
        self.project(x_to, v)
    }
}
