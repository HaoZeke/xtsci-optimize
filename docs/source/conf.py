#!/usr/bin/env python3

project = "xtsci-optimize"
copyright = "2024--present, Rohit Goswami"
author = "Rohit Goswami"
html_logo = "_static/xtsci-optimize-notext-light.webp"

extensions = [
    "myst_parser",
    "sphinx.ext.intersphinx",
]

templates_path = ["_templates"]
exclude_patterns = []

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
}

html_theme = "shibuya"
html_static_path = ["_static"]

html_theme_options = {
    "github_url": "https://github.com/HaoZeke/xtsci-optimize",
    "accent_color": "indigo",
    "dark_code": True,
    "globaltoc_expand_depth": 1,
    "light_logo": "_static/xtsci-optimize-notext-light.webp",
    "dark_logo": "_static/xtsci-optimize-notext-dark.webp",
    "nav_links": [
        {
            "title": "Ecosystem",
            "children": [
                {
                    "title": "eindir",
                    "url": "https://github.com/HaoZeke/eindir",
                    "summary": "Differentiable objectives",
                },
                {
                    "title": "rgpot",
                    "url": "https://github.com/OmniPotentRPC/rgpot",
                    "summary": "Potential-energy library and RPC server",
                },
                {
                    "title": "eOn",
                    "url": "https://eondocs.org",
                    "summary": "Saddle-point search on potential energy surfaces",
                },
            ],
        },
    ],
}

html_context = {
    "source_type": "github",
    "source_user": "HaoZeke",
    "source_repo": "xtsci-optimize",
    "source_version": "main",
    "source_docs_path": "/docs/source/",
}

html_sidebars = {
    "**": [
        "sidebars/localtoc.html",
        "sidebars/repo-stats.html",
        "sidebars/edit-this-page.html",
    ],
}

html_css_files = [
    "custom.css",
]
