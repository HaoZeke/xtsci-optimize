//! Wolfe line search on Rosenbrock; zoom stays inside its bracket.

use eindir_core::DifferentiableObjective;
use eindir_core::objectives::Rosenbrock;
use ndarray::{array, Array1, ArrayView1};
use xtsci_optimize::linesearch::zoom;
use xtsci_optimize::{Conjugacy, Control, LineSearch, Restart, minimize};

fn control() -> Control {
    Control {
        maxiter: 200,
        gtol: 1e-8,
        istep: 0.1,
        maxmove: None,
    }
}

fn wolfe() -> LineSearch {
    LineSearch::Wolfe {
        c1: 1e-4,
        c2: 0.9,
        maxiter: 40,
    }
}

/// `f(x) = (x[0] - 1)^2`. Line `x = α` along `d = 1` has minimizer `α = 1`.
fn quadratic(x: ArrayView1<'_, f64>) -> (f64, Array1<f64>) {
    let z = x[0] - 1.0;
    (z * z, array![2.0 * z])
}

#[test]
fn wolfe_on_rosenbrock_descends() {
    let obj = Rosenbrock::<2>::new();
    let start = array![-1.2, 1.0];
    let f0 = obj.value_and_gradient(start.view()).0;
    let report = minimize(
        &obj,
        start,
        &control(),
        Conjugacy::PolakRibiere,
        Restart::Never,
        wolfe(),
    )
    .unwrap();
    assert!(
        report.value < f0,
        "Wolfe {} -> {}",
        f0,
        report.value
    );
}

#[test]
fn zoom_stays_inside_the_bracket() {
    let pos = array![0.0];
    let dir = array![1.0];
    let lo = 0.0;
    let hi = 2.0;
    let alpha = zoom(
        &mut quadratic,
        pos.view(),
        dir.view(),
        lo,
        hi,
        1e-4,
        0.9,
        40,
    );
    assert!(
        alpha >= lo && alpha <= hi,
        "zoom left the bracket: {alpha} not in [{lo}, {hi}]"
    );
    let flipped = zoom(
        &mut quadratic,
        pos.view(),
        dir.view(),
        hi,
        lo,
        1e-4,
        0.9,
        40,
    );
    let a = lo.min(hi);
    let b = lo.max(hi);
    assert!(
        flipped >= a && flipped <= b,
        "zoom left the flipped bracket: {flipped} not in [{a}, {b}]"
    );
}

#[test]
fn goldstein_backtracking_descends() {
    let obj = Rosenbrock::<2>::new();
    let start = array![-1.2, 1.0];
    let f0 = obj.value_and_gradient(start.view()).0;
    let report = minimize(
        &obj,
        start,
        &control(),
        Conjugacy::PolakRibiere,
        Restart::Never,
        LineSearch::Goldstein {
            c: 1e-4,
            beta: 0.5,
            maxiter: 40,
        },
    )
    .unwrap();
    assert!(
        report.value < f0,
        "Goldstein {} -> {}",
        f0,
        report.value
    );
}
