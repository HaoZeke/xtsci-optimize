//! Persistent L-BFGS: watch hook, budget, warm start.

use ndarray::{Array1, ArrayView1};
use xtsci_optimize::Lbfgs;

fn quad(x: ArrayView1<f64>) -> (f64, Array1<f64>) {
    let scales = [1.0, 10.0, 100.0, 1000.0];
    let mut f = 0.0;
    let mut g = Array1::zeros(x.len());
    for i in 0..x.len() {
        let c = scales[i % scales.len()];
        f += 0.5 * c * x[i] * x[i];
        g[i] = c * x[i];
    }
    (f, g)
}

#[test]
fn watching_without_stopping_changes_nothing() {
    let x0 = Array1::from(vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
    let mut a = Lbfgs::default();
    let (fa, xa, ea) = a.minimize(x0.view(), 50, |v| Some(quad(v)));
    let mut b = Lbfgs::default();
    let (fb, xb, eb) = b.minimize_watched(x0.view(), 50, |v| Some(quad(v)), |_, _| true);
    assert_eq!(ea, eb);
    assert_eq!(fa, fb);
    assert_eq!(xa, xb);
}

#[test]
fn a_hook_that_refuses_stops_at_an_accepted_point() {
    let x0 = Array1::from(vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
    let mut opt = Lbfgs::default();
    let mut seen = Vec::new();
    let (f, _, _) = opt.minimize_watched(
        x0.view(),
        50,
        |v| Some(quad(v)),
        |it, fv| {
            seen.push(fv);
            it < 3
        },
    );
    assert_eq!(seen.len(), 4, "hook saw {} iterates", seen.len());
    assert_eq!(f, *seen.last().unwrap());
    assert!(f > 1e-10, "stopped hook still ran to convergence");
}

#[test]
fn converges_on_an_ill_conditioned_quadratic() {
    let mut opt = Lbfgs::default();
    let x0 = Array1::from(vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
    let (f, x, evals) = opt.minimize(x0.view(), 200, |v| Some(quad(v)));
    assert!(f < 1e-10, "did not converge, f = {f}, evals = {evals}");
    assert!(x.iter().all(|v| v.abs() < 1e-4));
}

#[test]
fn retained_curvature_costs_fewer_evaluations() {
    let x0 = Array1::from(vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);

    let mut warm = Lbfgs::default();
    let (_, xa, _) = warm.minimize(x0.view(), 200, |v| Some(quad(v)));
    let mut perturbed = xa.clone();
    for (i, v) in perturbed.iter_mut().enumerate() {
        *v += if i % 2 == 0 { 0.02 } else { -0.02 };
    }
    let (_, _, warm_evals) = warm.minimize(perturbed.view(), 200, |v| Some(quad(v)));

    let mut cold = Lbfgs::default();
    let (_, _, cold_evals) = cold.minimize(perturbed.view(), 200, |v| Some(quad(v)));

    assert!(
        warm_evals < cold_evals,
        "retained curvature should cost less: warm {warm_evals}, cold {cold_evals}"
    );
}

#[test]
fn forget_clears_the_memory() {
    let mut opt = Lbfgs::default();
    let x0 = Array1::from(vec![1.0, 1.0, 1.0, 1.0]);
    opt.minimize(x0.view(), 50, |v| Some(quad(v)));
    assert!(!opt.is_empty(), "a relaxation should store curvature");
    opt.forget();
    assert!(opt.is_empty());
}

#[test]
fn stops_when_the_budget_ends() {
    let mut opt = Lbfgs::default();
    let x0 = Array1::from(vec![1.0, 1.0, 1.0, 1.0]);
    let mut left = 3;
    let (_, _, evals) = opt.minimize(x0.view(), 200, |v| {
        if left == 0 {
            return None;
        }
        left -= 1;
        Some(quad(v))
    });
    assert!(evals <= 3, "spent {evals} with a budget of 3");
}
