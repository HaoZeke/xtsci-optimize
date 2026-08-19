# xtsci-optimize

Rust rewrite of [xtsci-optimize](https://github.com/HaoZeke/xtsci-optimize).
Algorithms live only in Rust, over
[`eindir`](https://github.com/HaoZeke/eindir) `DifferentiableObjective`s.

C and C++ keep the old `xts::optimize` names through an hourglass C ABI
(`xts_minimize`) that carries **dlpk** `DLManagedTensorVersioned` tensors.
`include/xts/optimize.hpp` is the C++ wrapper; `include/xts/xtensor.hpp`
adapts `xt::xarray`. There is no second solver implementation in C++.

CPU f64 is wired now. A non-CPU tensor returns `XTS_UNSUPPORTED_DEVICE`
so a CUDA path does not change the ABI.

Conjugacy: Fletcher-Reeves, Polak-Ribiere, Hestenes-Stiefel, Dai-Yuan,
conjugate descent, Hager-Zhang, Liu-Storey, FR-PR, HybridizedConj.
Line search: Brent, Armijo, Goldstein, strong Wolfe with zoom.
Methods: NLCG, BFGS, L-BFGS, SR1, SR2, Adam, steepest descent, PSO.

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
