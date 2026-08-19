//! Local first-order minimization over eindir objectives.
//!
//! Strategy split follows HaoZeke/xtsci-optimize: conjugacy, restart, and
//! line search are independent. Nonlinear CG is Nocedal and Wright
//! algorithm 5.4. Quasi-Newton methods (BFGS, L-BFGS, SR1) and Adam
//! share the same eindir `DifferentiableObjective` handle.

#![warn(missing_docs)]

/// Line-search strategies (Brent, Armijo backtracking).
pub mod linesearch;
/// Method selector (NLCG, BFGS family, Adam, steepest descent).
pub mod method;
/// Nonlinear conjugate-gradient conjugacy and restart.
pub mod nlcg;

mod adam;
mod control;
mod error;
mod minimize;
mod oracle;
mod qn;
mod report;
mod step;

pub use control::Control;
pub use error::{Error, Result};
pub use linesearch::LineSearch;
pub use method::Method;
pub use minimize::{minimize, minimize_method};
pub use nlcg::{Conjugacy, ConjugacyContext, Restart};
pub use oracle::Oracle;
pub use qn::{minimize_bfgs, minimize_lbfgs, minimize_sd, minimize_sr1};
pub use report::Report;
