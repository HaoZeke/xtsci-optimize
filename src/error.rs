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
}

/// Result alias for this crate.
pub type Result<T> = std::result::Result<T, Error>;
