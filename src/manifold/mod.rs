//! Embedded Riemannian manifolds (manopt_cpp / ROPTLIB waist).
//!
//! Absil, Mahony, Sepulchre, *Optimization Algorithms on Matrix
//! Manifolds*, <https://doi.org/10.1515/9781400830244>.
//! Boumal, *An Introduction to Optimization on Smooth Manifolds*,
//! <https://doi.org/10.1017/9781009166164>.
//! manopt_cpp: `proj`, `retr`, `transp` on an embedded Euclidean vector.

use ndarray::Array1;

mod euclidean;

pub use euclidean::Euclidean;

/// Which embedded geometry a session retracts onto.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ManifoldKind {
    /// Ambient Euclidean. Today's path.
    #[default]
    Euclidean,
    /// Unit sphere \(S^{n-1}\).
    Sphere,
    /// Rotation matrices SO(3), 9-vector row-major.
    So3,
    /// Stiefel \(\mathrm{St}(n,p)\). `p` from [`ManifoldKind::stiefel_p`].
    Stiefel,
    /// Rigid motions SE(3): 3x3 row-major then translation (12).
    Se3,
}

impl ManifoldKind {
    /// Stiefel column count. `p = 1` is the sphere.
    pub fn stiefel_p(self) -> usize {
        1
    }
}

/// manopt_cpp `AbstractManifold` on a rank-1 f64 vector.
pub trait Manifold {
    /// Tangent projection of an ambient vector at `x`.
    fn project(&self, x: &Array1<f64>, v: &Array1<f64>) -> Array1<f64>;
    /// Retraction of the tangent step `v` at `x`.
    fn retract(&self, x: &Array1<f64>, v: &Array1<f64>) -> Array1<f64>;
    /// Vector transport of `v` from `x_from` to `x_to`.
    fn transport(&self, x_from: &Array1<f64>, x_to: &Array1<f64>, v: &Array1<f64>) -> Array1<f64>;
}

impl Manifold for ManifoldKind {
    fn project(&self, x: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
        match self {
            Self::Euclidean => Euclidean.project(x, v),
            _ => Euclidean.project(x, v),
        }
    }

    fn retract(&self, x: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
        match self {
            Self::Euclidean => Euclidean.retract(x, v),
            _ => Euclidean.retract(x, v),
        }
    }

    fn transport(&self, x_from: &Array1<f64>, x_to: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
        match self {
            Self::Euclidean => Euclidean.transport(x_from, x_to, v),
            _ => Euclidean.transport(x_from, x_to, v),
        }
    }
}
