//! Which local method to run.

use crate::nlcg::{Conjugacy, Restart};

/// Local first-order, quasi-Newton, or swarm method over an eindir objective.
#[derive(Clone, Debug, PartialEq)]
pub enum Method {
    /// Nonlinear CG (Nocedal-Wright algorithm 5.4).
    Nlcg {
        /// β formula.
        conjugacy: Conjugacy,
        /// When to reset to steepest descent.
        restart: Restart,
    },
    /// Dense inverse-BFGS (Nocedal-Wright 6.17).
    Bfgs,
    /// Limited-memory BFGS two-loop recursion (Nocedal-Wright 7.4).
    Lbfgs {
        /// Correction pairs kept (`m`). Typical 5 to 20.
        memory: usize,
    },
    /// Inverse SR1 (Nocedal-Wright 6.24).
    Sr1,
    /// SR2 Hessian update (`xtsci-optimize` `sr2.hpp`).
    Sr2,
    /// Kingma and Ba Adam, with a line search along the preconditioned step.
    Adam {
        /// First-moment decay.
        beta1: f64,
        /// Second-moment decay.
        beta2: f64,
        /// Denominator floor.
        eps: f64,
    },
    /// Steepest descent: `d = -g` every step.
    Steepest,
    /// Particle swarm (xtsci `PSOptim`) over the objective box.
    Pso {
        /// Swarm size. Typical 10 to 40.
        n_particles: usize,
        /// Inertia weight `w`.
        inertia: f64,
        /// Cognitive coefficient.
        c1: f64,
        /// Social coefficient.
        c2: f64,
    },
}

impl Method {
    /// Polak-Ribiere CG, no restart. landfold `Solver::Standard`.
    pub fn polak_ribiere() -> Self {
        Self::Nlcg {
            conjugacy: Conjugacy::PolakRibiere,
            restart: Restart::Never,
        }
    }

    /// L-BFGS with `m = 10` correction pairs.
    pub fn lbfgs() -> Self {
        Self::Lbfgs { memory: 10 }
    }

    /// Adam with Kingma-Ba defaults.
    pub fn adam() -> Self {
        Self::Adam {
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
        }
    }

    /// PSO with xtsci defaults: 10 particles, `w = 0.5`, `c1 = c2 = 1.5`.
    pub fn pso() -> Self {
        Self::Pso {
            n_particles: 10,
            inertia: 0.5,
            c1: 1.5,
            c2: 1.5,
        }
    }

    /// NLCG with the given conjugacy and no restart.
    pub fn nlcg(conjugacy: Conjugacy) -> Self {
        Self::Nlcg {
            conjugacy,
            restart: Restart::Never,
        }
    }
}

impl Default for Method {
    fn default() -> Self {
        Self::polak_ribiere()
    }
}
