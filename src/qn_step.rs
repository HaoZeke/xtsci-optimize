//! How an L-BFGS session uses a caller-supplied Hessian.

/// eOn `lbfgs_step`. The pair / Lindh matrix stays with the caller.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum QnStep {
    /// Two-loop L-BFGS. A supplied Hessian is \(H_0 = P^{-1}\) (Packwood).
    #[default]
    TwoLoop,
    /// Regularized Newton on the supplied Hessian. Ignores stored pairs
    /// for the direction (eOn `lbfgs_step = newton`).
    Newton,
    /// Banerjee RFO on the supplied Hessian (eOn `lbfgs_step = rfo`).
    Rfo,
}
