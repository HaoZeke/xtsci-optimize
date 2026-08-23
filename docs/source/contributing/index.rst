

Contributing
------------

Build
~~~~~

Never compile on a laptop. rsync the checkout to the remote builder
and run ``cargo test --features capi`` there. ``eindir-core`` is a
pinned git dependency.

Citations
~~~~~~~~~

Rustdocs and the orgmode book may only cite DOIs on
`docs/CITATIONS.md <../../CITATIONS.md>`_ after ookcite ``verify_references``. Do not invent
DOIs.

Formulae
~~~~~~~~

Nocedal-Wright equation numbers are the contract:

- NLCG is algorithm 5.4

- Inverse BFGS is 6.17

- L-BFGS two-loop is 7.4, scale is 7.20

- Strong Wolfe is 3.5 / 3.6

- Goldstein is 3.11

- Hager-Zhang beta is paper equation 1.3, not the C++ port

Docs
~~~~

Narrative docs live in ``docs/orgmode/`` (Diataxis: tutorials, howto,
reference). Export to RST with ``emacs --script docs/export.el`` from
``docs/``. Sphinx (Shibuya) reads ``docs/source/``; logos live in
``branding/logo/`` and are copied into ``docs/source/_static/``.
The C ABI HTML is Doxygen plus doxyYoda; see
`howto/doxygen <../howto/doxygen.rst>`_.

Building the docs
-----------------

Export the orgmode tree to rst with ``emacs --batch -l docs/export.el``,
then build ``docs/source`` with Sphinx. The build needs ``shibuya``,
``myst-parser``, and ``sphinxcontrib-bibtex``; the bibliography comes from
``docs/source/references.bib``, which is generated from the
``rgmin-docs`` ookcite collection (``export_collection``), never edited by
hand. New citations enter through ookcite (``validate_doi`` first, then
``add_to_collection``) and land in ``docs/CITATIONS.md`` so rustdoc may
cite them.
