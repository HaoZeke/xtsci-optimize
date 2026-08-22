//! Stiefel \(\mathrm{St}(n,1)\): columns of an \(n\times p\) frame, \(p=1\) is the sphere.

use ndarray::Array1;

use super::{Manifold, sphere::Sphere};

/// Stiefel with \(p=1\) (sphere). Wider frames land on a later branch.
#[derive(Clone, Copy, Debug, Default)]
pub struct Stiefel;

impl Manifold for Stiefel {
    fn project(&self, x: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
        Sphere.project(x, v)
    }

    fn retract(&self, x: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
        Sphere.retract(x, v)
    }

    fn transport(&self, x_from: &Array1<f64>, x_to: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
        Sphere.transport(x_from, x_to, v)
    }
}
