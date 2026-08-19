//! Nocedal-Wright algorithm 5.4 on an eindir objective.

use eindir_core::{DifferentiableObjective, Objective};
use ndarray::Array1;

use crate::control::Control;
use crate::error::{Error, Result};
use crate::linesearch::LineSearch;
use crate::nlcg::{Conjugacy, ConjugacyContext, Restart};
use crate::report::Report;

/// Minimize `obj` from `init` with the chosen conjugacy, restart, and line search.
pub fn minimize<O>(
    obj: &O,
    init: impl Into<Array1<f64>>,
    control: &Control,
    conjugacy: Conjugacy,
    restart: Restart,
    linesearch: LineSearch,
) -> Result<Report>
where
    O: DifferentiableObjective<f64> + ?Sized,
{
    let mut pos = init.into();
    if pos.len() != Objective::dim(obj) {
        return Err(Error::Dim {
            got: pos.len(),
            dim: Objective::dim(obj),
        });
    }
    pos = obj.bounds().clip(pos.view());
    let (mut value, mut grad) = obj.value_and_gradient(pos.view());
    let mut dir = grad.mapv(|g| -g);
    let mut g_old = grad.clone();
    let mut d_old = dir.clone();
    let mut istep = control.istep;

    for step in 0..control.maxiter {
        let gnorm = l2(&grad);
        if gnorm < control.gtol {
            return Ok(Report {
                value,
                coords: pos,
                steps: step,
                grad_norm: gnorm,
            });
        }
        let (npos, nval, lsstep) = linesearch.search(
            |x| obj.value_and_gradient(x),
            pos.view(),
            dir.view(),
            istep,
        );
        let mut trial = obj.bounds().clip(npos.view());
        if let Some(cap) = control.maxmove {
            scale_step(&pos, &mut trial, cap);
        }
        if nval < value {
            pos = trial;
        }
        let ev = obj.value_and_gradient(pos.view());
        value = ev.0;
        grad = ev.1;
        let ctx = ConjugacyContext {
            current_gradient: grad.view(),
            previous_gradient: g_old.view(),
            previous_direction: d_old.view(),
        };
        let mut beta = conjugacy.beta(&ctx);
        if restart.should_restart(&ctx) {
            beta = 0.0;
        }
        dir = Array1::from_iter(
            grad.iter()
                .zip(d_old.iter())
                .map(|(g, d)| -g + beta * d),
        );
        g_old.assign(&grad);
        d_old.assign(&dir);
        istep = if lsstep <= 0.0 {
            control.istep
        } else {
            lsstep * 0.5
        };
    }
    Ok(Report {
        value,
        coords: pos,
        steps: control.maxiter,
        grad_norm: l2(&grad),
    })
}

fn l2(g: &Array1<f64>) -> f64 {
    g.iter().map(|x| x * x).sum::<f64>().sqrt()
}

fn scale_step(origin: &Array1<f64>, trial: &mut Array1<f64>, cap: f64) {
    let mut n2 = 0.0;
    for i in 0..trial.len() {
        let d = trial[i] - origin[i];
        n2 += d * d;
    }
    let n = n2.sqrt();
    if n > cap && n > 0.0 {
        let s = cap / n;
        for i in 0..trial.len() {
            trial[i] = origin[i] + s * (trial[i] - origin[i]);
        }
    }
}


