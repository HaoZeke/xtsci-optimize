

Constrain an IRC step to the mass-weighted sphere
-------------------------------------------------

The inner Gonzalez--Schlegel / Sella IRC step is not
``ManifoldKind::Sphere`` (the unit sphere about the origin). It is
an equality on ``MwRigid``:



.. math::

    \|(s + d_1)\odot\sqrt{m}\| = dx.

``IrcTrust`` is that restricted step.
``lowest_mode`` is the matrix-free kick (one extremal Hessian
pair from ``H v``, not a full ELPA / SLATE spectrum). Dispatch is
the closed ``EigensolverKind`` in ``schema/eigen.capnp``: Lanczos
(default), Rayleigh-Ritz, Jacobi-Davidson, LOBPCG. PRIMME,
SLEPc, ChASE, ELPA, ELPA2, SLATE, MAGMA, cuSOLVER, DLA-Future,
and EigenExa return ``Error::EigenUnavailable`` until linked.
There is no string key. The session that owns forward and reverse
branches lives in
`rgsaddle <https://github.com/OmniPotentRPC/rgsaddle>`_ (``IrcSession``).

Citations (validated DOIs, also in ``docs/CITATIONS.md``):

- Ishida, Morokuma, Komornicki 1977
  (`10.1063/1.434152 <https://doi.org/10.1063/1.434152>`_).

- Page and McIver 1988
  (`10.1063/1.454172 <https://doi.org/10.1063/1.454172>`_).

- Gonzalez and Schlegel 1989
  (`10.1063/1.456010 <https://doi.org/10.1063/1.456010>`_).

- Hermes, Sarsfield, Zador 2022
  (`10.1021/acs.jctc.2c00395 <https://doi.org/10.1021/acs.jctc.2c00395>`_).

Rust
~~~~

.. code:: rust

    use ndarray::array;
    use rgmin::{lowest_eigenpair, lowest_mode, EigenParams, EigensolverKind, IrcTrust};

    let masses = [1.0, 16.0];
    let d1 = array![0.1, 0.0, 0.0, 0.0, 0.0, 0.0];
    let tr = IrcTrust::from_atom_masses(d1, &masses, 0.2);
    let s = array![0.5, 0.1, -0.2, 0.3, 0.0, 0.4];
    let p = tr.project(&s);
    assert!(tr.on_bound(&p, 1e-12));

The kick uses ``lowest_eigenpair`` (Lanczos) or ``lowest_mode`` with
an ``EigenParams`` on a ``HessianVector`` (analytic or ``FdHvp``).
Default Krylov dimension is small. Do not form the dense Hessian
just to pick the imaginary mode. Unlinked backends fail closed.

What this is not
~~~~~~~~~~~~~~~~

- ``ManifoldKind::Sphere`` -- unit :math:`S^{n-1}` about 0.

- A full ``heev`` of a 3N Hessian. ELPA / SLATE belong behind
  ``n >= 512`` and a Hessian that already exists. See the rgsaddle
  MEP tutorial.
