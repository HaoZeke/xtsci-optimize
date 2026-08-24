

Bibliography
------------

Validated DOIs only. The Sphinx bibliography is
``docs/source/references.bib``, consumed by ``sphinxcontrib-bibtex``
(biblatex .bib, unsrt). New citations enter through ookcite
(``validate_doi``, then ``add_to_collection`` on ``rgmin-docs``), never
as invented keys or a markdown table. Regenerate the .bib with
``export_collection``. Narrative pages cite with the RST role
``:cite:`key```.

.. bibliography::
   :all:
