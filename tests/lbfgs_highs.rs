//! L-BFGS quadratic model through HiGHS.

#![cfg(feature = "highs")]

use approx::assert_relative_eq;
use ndarray::{Array1, ArrayView1};
use xtsci_optimize::{HighsStep, Lbfgs};

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
fn empty_memory_unbounded_step_is_steepest() {
    let opt = {
        let mut o = Lbfgs::default();
        o.highs = Some(HighsStep::default());
        o
    };
    let x = Array1::from(vec![1.0, 2.0, 3.0]);
    let g = Array1::from(vec![4.0, -5.0, 6.0]);
    let d = opt.highs_step(x.view(), g.view()).unwrap();
    assert_relative_eq!(d[0], -4.0, epsilon = 1e-6);
    assert_relative_eq!(d[1], 5.0, epsilon = 1e-6);
    assert_relative_eq!(d[2], -6.0, epsilon = 1e-6);
}

#[test]
fn unconstrained_qp_matches_two_loop() {
    let x = Array1::from(vec![1.0, 1.0, 1.0, 1.0]);
    let scales = [1.0, 10.0, 100.0, 1000.0];
    let s = Array1::from(vec![-0.1, -0.1, -0.1, -0.1]);
    let y = Array1::from_iter((0..4).map(|i| scales[i] * s[i]));
    let mut warm = Lbfgs::default();
    warm.record(s, y);
    let g = quad(x.view()).1;
    let two = warm.two_loop(g.view());
    warm.highs = Some(HighsStep::default());
    let qp = warm.highs_step(x.view(), g.view()).unwrap();
    for i in 0..two.len() {
        assert_relative_eq!(qp[i], two[i], epsilon = 1e-5, max_relative = 1e-4);
    }
}

#[test]
fn box_keeps_the_trial_inside() {
    let mut opt = Lbfgs::default();
    opt.highs = Some(HighsStep {
        trust: Some(0.25),
        lo: Some(-0.3),
        hi: Some(0.3),
        equalities: Vec::new(),
        center_axes: None,
    });
    let x = Array1::from(vec![0.2, -0.2]);
    let g = Array1::from(vec![10.0, -10.0]);
    let d = opt.highs_step(x.view(), g.view()).unwrap();
    for i in 0..x.len() {
        let t = x[i] + d[i];
        assert!(t >= -0.3 - 1e-9 && t <= 0.3 + 1e-9, "left the box: {t}");
        assert!(d[i].abs() <= 0.25 + 1e-9, "left the trust: {}", d[i]);
    }
}

#[test]
fn highs_lbfgs_converges_on_the_quadratic() {
    let mut opt = Lbfgs::default();
    opt.highs = Some(HighsStep::default());
    let x0 = Array1::from(vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
    let (f, x, evals) = opt.minimize(x0.view(), 80, |v| Some(quad(v)));
    assert!(f < 1e-8, "did not converge, f = {f}, evals = {evals}");
    assert!(x.iter().all(|v| v.abs() < 1e-3));
}

#[test]
fn equality_projection_scales_to_thirty_two_variables() {
    let mut opt = Lbfgs::default();
    opt.highs = Some(HighsStep {
        equalities: vec![(vec![(0, 1.0), (1, 1.0)], 0.0)],
        ..HighsStep::default()
    });
    let x0 = Array1::from_elem(32, 1.0);
    let (_, x, _) = opt.minimize(x0.view(), 4, |v| Some(quad(v)));
    assert!(x.iter().all(|value| value.is_finite()));
    assert!((x[0] + x[1]).abs() < 1e-8);
}

#[test]
fn trust_scale_keeps_the_two_loop_direction() {
    let x = Array1::from(vec![0.0, 0.0]);
    let g = Array1::from(vec![10.0, 1.0]);
    let mut opt = Lbfgs::default();
    let raw = opt.two_loop(g.view());
    opt.highs = Some(HighsStep {
        trust: Some(0.25),
        lo: None,
        hi: None,
        equalities: Vec::new(),
        center_axes: None,
    });
    let p = opt.highs_step(x.view(), g.view()).unwrap();
    let n2 = raw.dot(&raw).sqrt();
    let np = p.dot(&p).sqrt();
    let pinf = p.iter().fold(0.0_f64, |a, v| a.max(v.abs()));
    assert!(pinf <= 0.25 + 1e-12);
    for i in 0..2 {
        assert_relative_eq!(p[i] / np, raw[i] / n2, epsilon = 1e-12);
    }
}

#[test]
fn center_axes_kills_the_mean() {
    let mut opt = Lbfgs::default();
    opt.highs = Some(HighsStep {
        trust: Some(0.5),
        lo: None,
        hi: None,
        equalities: Vec::new(),
        center_axes: Some((4, 2)),
    });
    let x = Array1::zeros(8);
    let g = Array1::from(vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]);
    let p = opt.highs_step(x.view(), g.view()).unwrap();
    let mut mx = 0.0;
    let mut my = 0.0;
    for i in 0..4 {
        mx += p[i * 2];
        my += p[i * 2 + 1];
    }
    assert!(mx.abs() < 1e-12, "axis 0 mean {mx}");
    assert!(my.abs() < 1e-12, "axis 1 mean {my}");
    for v in p.iter() {
        assert!(v.abs() <= 0.5 + 1e-12);
    }
}
