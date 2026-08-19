//! Iteration limits and gradient tolerance.

/// Outer-loop controls for [`crate::minimize`].
#[derive(Clone, Debug)]
pub struct Control {
    /// Maximum CG iterations.
    pub maxiter: usize,
    /// Stop when `||g||_2 < gtol`.
    pub gtol: f64,
    /// Initial line-search step guess.
    pub istep: f64,
    /// Optional Euclidean cap on a proposed step (xtsci `maxmove`).
    pub maxmove: Option<f64>,
}

impl Default for Control {
    fn default() -> Self {
        Self {
            maxiter: 100,
            gtol: 1e-5,
            istep: 1.0,
            maxmove: None,
        }
    }
}
