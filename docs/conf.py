"""Sphinx configuration for contextkit API documentation."""

import os
import sys

# -- Path setup -------------------------------------------------------
sys.path.insert(0, os.path.abspath("../src"))

# -- Project information -----------------------------------------------
project = "contextkit"
copyright = "2025, Royce"
author = "Royce"
release = "0.1.0"

# -- General configuration ---------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "autoapi.extension",
]

# -- sphinx-autoapi configuration --------------------------------------
# Automatically generates API docs from source code — no manual .rst files
# needed per module.
autoapi_dirs = ["../src/contextkit"]
autoapi_type = "python"
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
]
autoapi_ignore = ["*/tests/*", "*/__pycache__/*"]
autoapi_keep_files = True
autoapi_add_toctree_entry = True
autoapi_python_class_content = "both"
autoapi_member_order = "groupwise"

# -- Napoleon (Google-style docstrings) --------------------------------
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_use_param = True
napoleon_use_rtype = True

# -- Intersphinx (cross-reference Python stdlib) -----------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "pydantic": ("https://docs.pydantic.dev/latest/", None),
}

# Don't fail the build on unreachable intersphinx inventories.
suppress_warnings = ["intersphinx.external"]

# -- HTML output -------------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "navigation_depth": 4,
    "collapse_navigation": False,
    "sticky_navigation": True,
}

# -- Exclude patterns --------------------------------------------------
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Templates ---------------------------------------------------------
templates_path = ["_templates"]
