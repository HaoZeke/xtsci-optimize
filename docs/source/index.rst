
===============
rgmin
===============

:Author: `Rohit Goswami <https://rgoswami.me>`_

.. raw:: html

   <p class="rgmin-hero">
     <img class="rgmin-hero-logo rgmin-hero-logo--light"
          src="_static/rgmin-logo-light.webp"
          alt="rgmin logo"
          width="420"
          height="84"
          loading="eager" />
     <img class="rgmin-hero-logo rgmin-hero-logo--dark"
          src="_static/rgmin-logo-dark.webp"
          alt="rgmin logo"
          width="420"
          height="84"
          loading="eager" />
   </p>

Overview
--------

``rgmin`` is the local-minimization crate of the OmniPotentRPC family, beside `rgpot <https://github.com/OmniPotentRPC/rgpot>`_; it was xtsci-optimize until the xtensor heritage stopped being true.
Algorithms live only in Rust, over
`eindir <https://github.com/HaoZeke/eindir>`_ ~DifferentiableObjective~s.

C and C++ keep the old ``xts::optimize`` names through an hourglass C ABI
(``rgmin_solver_t`` / ``rgmin_minimize``) that carries **dlpk**
``DLManagedTensorVersioned`` tensors.
``include/xts/optimize.hpp`` is the C++ wrapper; ``include/xts/xtensor.hpp``
adapts ``xt::xarray``. There is no second solver implementation in C++.

The C ABI also accepts an ``eindir_objective_t*`` through
``rgmin_minimize_eindir``. The caller supplies the ``eindir_abi_stamp_t``,
retains ownership of the objective, and gets the same CPU f64 DLPack
path. That is how rgpot's first-member ``eindir_objective_t`` embedding
reaches the solvers without a callback shim in each consumer.

The production unconstrained local method is limited-memory BFGS with
the strong Wolfe conditions (Nocedal and Wright 7.4 and algorithms
3.5--3.6). Hopping chains keep that solver's pair history on ``Lbfgs``;
anneal's ``WarmLbfgs`` is a thin handle around it.

.. code:: bash

    git clone https://github.com/OmniPotentRPC/rgmin.git
    cd rgmin
    # Build and test on the remote builder, not a laptop.
    cargo test --features capi

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   tutorials/quickstart

.. toctree::
   :maxdepth: 2
   :caption: How-To Guides

   howto/session
   howto/manifolds
   howto/embed-anneal
   howto/highs-qp
   howto/c-abi
   howto/doxygen

.. toctree::
   :maxdepth: 2
   :caption: Explanation

   explanation/linesearch
   explanation/quasi-newton
   explanation/trust-region
   explanation/scg
   explanation/randomized

.. toctree::
   :maxdepth: 2
   :caption: Reference

   reference/architecture
   reference/sota
   reference/proofs
   reference/bibliography
   changelog

.. toctree::
   :maxdepth: 2
   :caption: Development

   contributing/index

License
-------

MIT.
