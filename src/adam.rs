//! Kingma and Ba Adam, with a line search along the preconditioned step.

use eindir_core::{DifferentiableObjective, Objective};
use ndarray::Array1;

use crate::control::Control;
use crate::error::{Error, Result};
use crate::linesearch::LineSearch;
use crate::report::Report;
use crate::step::{l2, next_istep, take_step};

/// Adam (Kingma and Ba, ICLR 2015) plus Brent/Armijo along `-mhat / (sqrt(vhat)+eps)`.
pub fn minimize_adam<O>(
    obj: &O,
    init: impl Into<Array1<f64>>,
    control: &Control,
    linesearch: LineSearch,
    beta1: f64,
    beta2: f64,
    eps: f64,
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
    let mut m = Array1::<f64>::zeros(pos.len());
    let mut v = Array1::<f64>::zeros(pos.len());
    let mut b1p = beta1;
    let mut b2p = beta2;
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
        for i in 0..pos.len() {
            m[i] = beta1 * m[i] + (1.0 - beta1) * grad[i];
            v[i] = beta2 * v[i] + (1.0 - beta2) * grad[i] * grad[i];
        }
        let mut dir = Array1::<f64>::zeros(pos.len());
        for i in 0..pos.len() {
            let mhat = m[i] / (1.0 - b1p);
            let vhat = v[i] / (1.0 - b2p);
            dir[i] = -mhat / (vhat.sqrt() + eps);
        }
        let (npos, _, lsstep, _) =
            take_step(obj, &pos, value, dir.view(), istep, linesearch, control);
        pos = npos;
        let ev = obj.value_and_gradient(pos.view());
        value = ev.0;
        grad = ev.1;
        b1p *= beta1;
        b2p *= beta2;
        istep = next_istep(lsstep, control);
    }
    Ok(Report {
        value,
        coords: pos,
        steps: control.maxiter,
        grad_norm: l2(&grad),
    })
}
