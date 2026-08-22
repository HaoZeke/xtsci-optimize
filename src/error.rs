//! Errors from a minimization.

use thiserror::Error;

/// Recoverable minimization failure.
#[derive(Debug, Error)]
pub enum Error {
    /// Initial point length does not match the objective dimension.
    #[error("init length {got} != objective dim {dim}")]
    Dim {
        /// Length of the supplied start vector.
        got: usize,
        /// `Objective::dim`.
        dim: usize,
    },
    /// HiGHS rejected the L-BFGS quadratic model.
    #[error("HiGHS: {0}")]
    Highs(String),
    /// Newton / RFO / dogleg needs a Hessian oracle.
    #[error("Newton/RFO/dogleg needs a Hessian; call step_hess")]
    NeedHessian,
    /// Packed manifold rejected this ambient dimension.
    #[error("{kind} rejected dimension {got}")]
    ManifoldDim {
        /// Token (`so3`, `se3`, `rigid_quotient`, `mw_rigid`).
        kind: &'static str,
        /// Length of the working vector.
        got: usize,
    },
}

/// Result alias for this crate.
pub type Result<T> = std::result::Result<T, Error>;
