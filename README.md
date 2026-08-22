# xtsci-optimize

Rust rewrite. `main` is this tree (first tag `v0.1.0`). The C++
xtensor history is `0.0.1` on the previous `main`.

Algorithms live only in Rust, over
Algorithms live only in Rust, over
[`eindir`](https://github.com/HaoZeke/eindir) `DifferentiableObjective`s.

C and C++ keep the old `xts::optimize` names through an hourglass C ABI
(`xts_minimize`) that carries **dlpk** `DLManagedTensorVersioned` tensors.
`include/xts/optimize.hpp` is the C++ wrapper; `include/xts/xtensor.hpp`
adapts `xt::xarray`. There is no second solver implementation in C++.

The C ABI also accepts an `eindir_objective_t*` directly through
`xts_minimize_eindir`. The caller supplies the `eindir_abi_stamp_t`, retains
ownership of the objective, and gets the same CPU f64 DLPack optimization path.
This lets rgpot's first-member `eindir_objective_t` embedding reach xtsci
without a callback shim in each consumer.

CPU f64 is wired now. A non-CPU tensor returns `XTS_UNSUPPORTED_DEVICE`
so a CUDA path does not change the ABI.

The production unconstrained local method is [`Lbfgs`] with strong Wolfe
(Nocedal-Wright 7.4 and algorithms 3.5 / 3.6). anneal `WarmLbfgs` holds
that type so hopping does not ship a second two-loop. Feature `highs` keeps the two-loop direction and projects it with
HiGHS (`Q = I`) onto a box, trust region, or equalities.

Conjugacy: Fletcher-Reeves, Polak-Ribiere, Hestenes-Stiefel, Dai-Yuan,
conjugate descent, Hager-Zhang, Liu-Storey, FR-PR, HybridizedConj.
Line search: Brent, Armijo, Goldstein, strong Wolfe with zoom.
Methods: NLCG, BFGS, L-BFGS, SR1, SR2, Newton, RFO, Adam, steepest
descent, PSO, FIRE, FIRE 2.0, Barzilai-Borwein, Powell dogleg.

Narrative docs live in [`docs/orgmode/`](docs/orgmode/index.org)
(Diataxis). C ABI HTML is Doxygen plus
[doxyYoda](https://github.com/HaoZeke/doxyYoda):
`scripts/get_doxyyoda.sh` then `doxygen docs/source/Doxyfile_doxyyoda.cfg`.
Validated DOIs are in [`docs/CITATIONS.md`](docs/CITATIONS.md).

```rust
use eindir_core::objectives::Rosenbrock;
use ndarray::array;
use xtsci_optimize::{Conjugacy, Control, LineSearch, Restart, minimize};

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

MIT. Build and test on the remote builder, not a laptop.
