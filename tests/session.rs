//! Persistent Session: one step is one outer iteration.

use eindir_core::objectives::Rosenbrock;
use ndarray::array;
use xtsci_optimize::{Control, Method, Solver};

fn control() -> Control {
    Control {
        maxiter: 80,
        gtol: 1e-8,
        istep: 0.1,
        maxmove: None,
    }
}

#[test]
fn lbfgs_session_reaches_rosenbrock() {
    let obj = Rosenbrock::<2>::new();
    let mut x = array![-1.2, 1.0];
    let mut solver = Solver::new(Method::lbfgs(), control(), 2).with_gtol(1e-8);
    let mut last = None;
    for _ in 0..80 {
        let rep = solver.step(&obj, &mut x).unwrap();
        last = Some(rep);
        if last.as_ref().unwrap().grad_norm < 1e-8 {
            break;
        }
    }
    let rep = last.unwrap();
    assert!(rep.value < 1e-6, "value {}", rep.value);
    assert!((x[0] - 1.0).abs() < 1e-3);
    assert!((x[1] - 1.0).abs() < 1e-3);
}

#[test]
fn retained_pairs_beat_a_cold_start() {
    let obj = Rosenbrock::<2>::new();
    let mut warm = Solver::new(Method::lbfgs(), control(), 2).with_gtol(1e-8);
    let mut x = array![-1.2, 1.0];
    for _ in 0..80 {
        let rep = warm.step(&obj, &mut x).unwrap();
        if rep.grad_norm < 1e-8 {
            break;
        }
    }
    x[0] += 0.05;
    x[1] -= 0.05;
    let mut warm_steps = 0usize;
    for _ in 0..80 {
        warm_steps += 1;
        let rep = warm.step(&obj, &mut x).unwrap();
        if rep.grad_norm < 1e-8 {
            break;
        }
    }

    let mut cold = Solver::new(Method::lbfgs(), control(), 2).with_gtol(1e-8);
    let mut y = array![-1.2 + 0.05, 1.0 - 0.05];
    // Same start as the warm restart: first quench, then same perturbation.
    let mut z = array![-1.2, 1.0];
    for _ in 0..80 {
        let rep = cold.step(&obj, &mut z).unwrap();
        if rep.grad_norm < 1e-8 {
            break;
        }
    }
    cold.forget();
    y = z;
    y[0] += 0.05;
    y[1] -= 0.05;
    let mut cold_steps = 0usize;
    for _ in 0..80 {
        cold_steps += 1;
        let rep = cold.step(&obj, &mut y).unwrap();
        if rep.grad_norm < 1e-8 {
            break;
        }
    }
    assert!(
        warm_steps <= cold_steps,
        "warm {warm_steps} should not exceed cold {cold_steps}"
    );
}

#[test]
#[test]
fn lbfgs_newton_on_a_supplied_hessian_kills_a_quadratic() {
    use eindir_core::{Bounds, DifferentiableObjective, Gradient, Objective};
    use ndarray::{Array2, ArrayView1};
    use xtsci_optimize::{HessianObjective, QnStep};

    struct Quad;
    impl Objective<f64> for Quad {
        fn dim(&self) -> usize {
            2
        }
        fn bounds(&self) -> &Bounds<f64> {
            use std::sync::OnceLock;
            static B: OnceLock<Bounds<f64>> = OnceLock::new();
            B.get_or_init(|| {
                Bounds::new(array![-1e6, -1e6], array![1e6, 1e6], 0.0)
            })
        }
        fn eval(&self, x: ArrayView1<f64>) -> f64 {
            5.0 * x[0] * x[0] + 0.5 * x[1] * x[1]
        }
    }
    impl Gradient<f64> for Quad {
        fn dim(&self) -> usize {
            2
        }
        fn grad(&self, x: ArrayView1<f64>) -> ndarray::Array1<f64> {
            array![10.0 * x[0], x[1]]
        }
    }
    impl DifferentiableObjective<f64> for Quad {
        fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, ndarray::Array1<f64>) {
            (self.eval(x), self.grad(x))
        }
    }
    impl HessianObjective for Quad {
        fn hessian(&self, _x: ArrayView1<f64>) -> Array2<f64> {
            Array2::from_shape_vec((2, 2), vec![10.0, 0.0, 0.0, 1.0]).unwrap()
        }
    }

    let obj = Quad;
    let mut x = array![2.0, -3.0];
    let mut solver = Solver::new(Method::lbfgs(), control(), 2).with_gtol(1e-10);
    solver.set_qn_step(QnStep::Newton);
    let rep = solver.step_hess(&obj, &mut x).unwrap();
    assert!(rep.value < 1e-12, "newton-on-P value {}", rep.value);
    assert!(x.iter().all(|v| v.abs() < 1e-6));
}

fn nlcg_second_step_is_not_steepest() {
    let obj = Rosenbrock::<2>::new();
    let mut a = array![-1.2, 1.0];
    let mut b = array![-1.2, 1.0];
    let mut pr = Solver::new(Method::polak_ribiere(), control(), 2);
    let mut sd = Solver::new(Method::Steepest, control(), 2);
    let _ = pr.step(&obj, &mut a).unwrap();
    let _ = sd.step(&obj, &mut b).unwrap();
    let _ = pr.step(&obj, &mut a).unwrap();
    let _ = sd.step(&obj, &mut b).unwrap();
    // After two steps the conjugacy history must move PR off the SD path.
    assert!(
        (a[0] - b[0]).abs() + (a[1] - b[1]).abs() > 1e-12,
        "PR and steepest stayed together: {a:?} {b:?}"
    );
}
