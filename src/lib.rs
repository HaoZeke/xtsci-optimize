//! Local first-order minimization over eindir objectives.
//!
//! This crate is the Rust rewrite of HaoZeke/xtsci-optimize. Conjugacy,
//! restart, and line search stay independent. Nonlinear CG is Nocedal and
//! Wright algorithm 5.4. Quasi-Newton methods (BFGS, L-BFGS, SR1, SR2),
//! Adam, steepest descent, and particle swarm share the same eindir
//! `DifferentiableObjective` handle. C and C++ reach these solvers only
//! through `xts_minimize` and the headers under `include/`.

#![warn(missing_docs)]

/// Line-search strategies (Brent, Armijo, Goldstein, Wolfe/zoom).
pub mod linesearch;
/// Method selector (NLCG, BFGS family, Adam, steepest descent, PSO).
pub mod method;
/// Nonlinear conjugate-gradient conjugacy and restart.
pub mod nlcg;

mod adam;
mod control;
mod error;
/// C ABI, gated behind the `capi` feature.
#[cfg(feature = "capi")]
#[allow(non_camel_case_types, missing_docs)]
pub mod ffi;
mod minimize;
mod oracle;
mod pso;
mod qn;
mod report;
mod step;

pub use control::Control;
pub use error::{Error, Result};
pub use linesearch::LineSearch;
pub use method::Method;
pub use minimize::{minimize, minimize_method};
pub use nlcg::{Conjugacy, ConjugacyContext, Restart};
pub use adam::minimize_adam;
pub use oracle::Oracle;
pub use pso::minimize_pso;
pub use qn::{minimize_bfgs, minimize_lbfgs, minimize_sd, minimize_sr1, minimize_sr2};
pub use report::Report;
