//! Local first-order minimization over eindir objectives.
//!
//! Strategy split follows HaoZeke/xtsci-optimize: conjugacy, restart, and
//! line search are independent. The loop is Nocedal and Wright algorithm
//! 5.4 (direction `d = -g + β d_prev`).

#![warn(missing_docs)]

/// Line-search strategies (Brent, Armijo backtracking).
pub mod linesearch;
/// Nonlinear conjugate-gradient conjugacy and restart.
pub mod nlcg;

mod control;
mod error;
mod minimize;
mod oracle;
mod report;

pub use control::Control;
pub use error::{Error, Result};
pub use linesearch::LineSearch;
pub use minimize::minimize;
pub use nlcg::{Conjugacy, ConjugacyContext, Restart};
pub use oracle::Oracle;
pub use report::Report;
