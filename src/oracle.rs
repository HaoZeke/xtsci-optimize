//! Closure adapter: `(x) -> (f, grad)` as an eindir objective.

use eindir_core::{Bounds, DifferentiableObjective, Gradient, Objective};
use ndarray::{Array1, ArrayView1};

/// Wraps a fused `(value, gradient)` closure as [`DifferentiableObjective`].
pub struct Oracle<F>
where
    F: Fn(ArrayView1<f64>) -> (f64, Array1<f64>) + Send + Sync,
{
    f: F,
    bounds: Bounds<f64>,
}

impl<F> Oracle<F>
where
    F: Fn(ArrayView1<f64>) -> (f64, Array1<f64>) + Send + Sync,
{
    /// Wide-box oracle of dimension `dim`.
    pub fn unbounded(dim: usize, f: F) -> Self {
        const LO: f64 = -1e12;
        const HI: f64 = 1e12;
        Self {
            f,
            bounds: Bounds::new(Array1::from_elem(dim, LO), Array1::from_elem(dim, HI), 0.0),
        }
    }
}

impl<F> Objective<f64> for Oracle<F>
where
    F: Fn(ArrayView1<f64>) -> (f64, Array1<f64>) + Send + Sync,
{
    fn dim(&self) -> usize {
        self.bounds.dims
    }

    fn bounds(&self) -> &Bounds<f64> {
        &self.bounds
    }

    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        (self.f)(x).0
    }
}

impl<F> Gradient<f64> for Oracle<F>
where
    F: Fn(ArrayView1<f64>) -> (f64, Array1<f64>) + Send + Sync,
{
    fn dim(&self) -> usize {
        self.bounds.dims
    }

    fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
        (self.f)(x).1
    }
}

impl<F> DifferentiableObjective<f64> for Oracle<F>
where
    F: Fn(ArrayView1<f64>) -> (f64, Array1<f64>) + Send + Sync,
{
    fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        (self.f)(x)
    }
}
