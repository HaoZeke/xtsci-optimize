//! PSO on Rosenbrock 2D inside the builtin box, swarm RNG seed 1.

use eindir_core::DifferentiableObjective;
use eindir_core::objectives::Rosenbrock;
use ndarray::array;
use quench_core::{Control, LineSearch, Method, minimize_method};

#[test]
fn pso_reports_below_classic_rosenbrock_start() {
    let obj = Rosenbrock::<2>::new();
    let start = array![-1.2, 1.0];
    let f0 = obj.value_and_gradient(start.view()).0;
    let report = minimize_method(
        &obj,
        start,
        &Control {
            maxiter: 200,
            gtol: 1e-8,
            istep: 0.1,
            maxmove: None,
        },
        Method::pso(),
        LineSearch::default(),
    )
    .unwrap();
    assert!(
        report.value < f0,
        "PSO {} -> {}",
        f0,
        report.value
    );
}
