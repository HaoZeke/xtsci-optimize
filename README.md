# quench

`quench-core` is the Rust rewrite of
[xtsci-optimize](https://github.com/HaoZeke/xtsci-optimize). Algorithms
live only in Rust. C and C++ keep ABI compatibility through
`quench_minimize_fn`, `include/quench.hpp`, and
`include/quench/xtensor.hpp`; `include/xts/optimize.hpp` maps the old
`xts::optimize` names onto that hourglass.

Local first-order minimization over
[`eindir`](https://github.com/HaoZeke/eindir) `DifferentiableObjective`s.

`eindir-core` owns the typed `S -> R` handle. `anneal-core` owns the
five-slot simulated-annealing algebra. This crate is the local quench:
nonlinear conjugate gradient with interchangeable conjugacy, restart,
and line search.

Conjugacy methods follow Nocedal and Wright, *Numerical Optimization*
(2006), chapter 5: Fletcher-Reeves, Polak-Ribiere, Hestenes-Stiefel,
Dai-Yuan, conjugate descent, Hager-Zhang, Liu-Storey, the FR-PR hybrid,
and HybridizedConj (`max`/`min` of two betas). Line search is Brent
(1973), Armijo backtracking, Goldstein, or strong Wolfe with
Nocedal-Wright zoom. Quasi-Newton
methods (BFGS, L-BFGS, SR1, SR2), Adam, steepest descent, and particle
swarm share the same `Method` enum and eindir handle.

```rust
use eindir_core::objectives::Rosenbrock;
use ndarray::array;
use quench_core::{Conjugacy, Control, LineSearch, Restart, minimize};

let obj = Rosenbrock::<2>::new();
let report = minimize(
    &obj,
    array![-1.2, 1.0],
    &Control::default(),
    Conjugacy::PolakRibiere,
    Restart::Never,
    LineSearch::Brent { maxiter: 40, tol: 1e-10 },
);
```

Hourglass (metatensor-shaped): Rust is the only implementation. The
`capi` feature exports `quench_minimize_fn` over **dlpk**
(`DLManagedTensorVersioned`), the same waist as eindir. CPU f64 is
wired now; a non-CPU tensor returns `QUENCH_UNSUPPORTED_DEVICE` so a
CUDA path does not change the ABI. `include/quench.hpp` wraps the C
API; `include/quench/xtensor.hpp` borrows `xt::xarray` as DLPack;
`include/xts/optimize.hpp` keeps `xts::optimize` names as Method tags.

MIT. Build and test on the remote builder, not a laptop.
