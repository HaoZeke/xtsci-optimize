======================================================
Scaled conjugate gradients: a trust region in disguise
======================================================


.. contents::

Moller's scaled conjugate gradient (``src/scg.rs``) is the odd member of
the family: a conjugate-gradient method with no line search at all.
It is worth understanding as a one-dimensional trust region, because
that is what its damping parameter is.

1 The step from a damped model
------------------------------

Along the current conjugate direction ``d``, SCG builds the quadratic
model from the directional derivative mu = d . g and a curvature
estimate gamma = d . H d, then damps the curvature with lambda >= 0:

delta = gamma + lambda |d|\ :sup:`2`\,     alpha = -mu / delta.

The damping plays exactly the role of the trust radius: lambda = 0
trusts the quadratic model fully (a Newton-like step along the ray);
large lambda shrinks the step toward a small gradient step. After each
step the comparison ratio

Delta = 2 (f(x + alpha d) - f(x)) / (alpha mu)

measures how quadratic the function actually was, and lambda is
lowered when the model earned trust (Delta large) and raised
fourfold when it did not -- the same shrink-grow policy as
``update_radius``, acting on stiffness instead of length.

2 Where the curvature comes from
--------------------------------

The reference algorithm prices gamma with a one-sided finite
difference of gradients along ``d``, one extra gradient per iteration.
When the objective implements ``DirectionalCurvature`` -- and via the
blanket impl, whenever it implements ``HessianVector`` -- the crate's
``minimize_scg_exact`` replaces the probe with the exact action
d . H(x) d, the same Newton-Krylov pricing TAO's ``nls`` obtains from
user Hessian-vector callbacks. One trait method turns the probe cost
into an action cost; nothing else in the algorithm changes.

3 Safeguards
------------

A non-descent direction (mu >= 0) resets to steepest descent before
the step is priced -- conjugacy is a performance device, never a
correctness assumption. Non-finite trial values raise the damping and
retry within a fixed budget, and a probe that cannot find a finite
curvature reports ``ScgStalled`` rather than inventing one. The
convergence gate combines a step-size test in the infinity norm with
a relative objective test, both scaled, so neither a tiny step at a
steep point nor a flat stretch at a large step reads as convergence
alone.
