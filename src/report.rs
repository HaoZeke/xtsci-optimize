//! Result of a quench.

use ndarray::Array1;

/// Final point, value, and iteration count.
#[derive(Clone, Debug)]
pub struct Report {
    /// `f(x)` at [`Report::coords`].
    pub value: f64,
    /// Accepted coordinates.
    pub coords: Array1<f64>,
    /// Outer iterations performed.
    pub steps: usize,
    /// `||∇f||_2` at the accepted point.
    pub grad_norm: f64,
}
