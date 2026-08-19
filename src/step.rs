//! Shared accept / clip / scale for one line-search move.

use eindir_core::DifferentiableObjective;
use ndarray::{Array1, ArrayView1};

use crate::control::Control;
use crate::linesearch::LineSearch;

pub(crate) fn l2(g: &Array1<f64>) -> f64 {
    g.iter().map(|x| x * x).sum::<f64>().sqrt()
}

pub(crate) fn scale_step(origin: &Array1<f64>, trial: &mut Array1<f64>, cap: f64) {
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

/// Line search, clip to bounds, optional maxmove. Returns `(x, f, |α|, moved)`.
pub(crate) fn take_step<O>(
    obj: &O,
    pos: &Array1<f64>,
    value: f64,
    dir: ArrayView1<'_, f64>,
    istep: f64,
    linesearch: LineSearch,
    control: &Control,
) -> (Array1<f64>, f64, f64, bool)
where
    O: DifferentiableObjective<f64> + ?Sized,
{
    let (npos, nval, lsstep) =
        linesearch.search(|x| obj.value_and_gradient(x), pos.view(), dir, istep);
    let mut trial = obj.bounds().clip(npos.view());
    if let Some(cap) = control.maxmove {
        scale_step(pos, &mut trial, cap);
    }
    if nval < value {
        (trial, nval, lsstep, true)
    } else {
        (pos.clone(), value, 0.0, false)
    }
}

pub(crate) fn next_istep(lsstep: f64, control: &Control) -> f64 {
    if lsstep <= 0.0 {
        control.istep
    } else {
        lsstep * 0.5
    }
}
