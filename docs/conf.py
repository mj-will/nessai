import os
import subprocess
import sys

import nessai

# -- Run pre-build steps script ----------------------------------------------


def run_external_script(script_path):
    """Run a script to generate plots."""
    script_path = os.path.abspath(script_path)
    subprocess.run(["python", script_path], check=True)


# Generate plots from an example run
run_external_script("scripts/example_run.py")

# -- Path setup --------------------------------------------------------------

sys.path.insert(0, os.path.abspath("../src/nessai/"))

# -- Project information -----------------------------------------------------

project = "nessai"
copyright = "2020, Michael J. Williams"
author = "Michael J. Williams"
version = nessai.__version__
release = nessai.__version__

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.mathjax",
    "numpydoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.autodoc",
    "sphinx.ext.inheritance_diagram",
    "sphinx_tabs.tabs",
    "autoapi.extension",
    "IPython.sphinxext.ipython_console_highlighting",
    "IPython.sphinxext.ipython_directive",
]
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_book_theme"
html_static_path = []
html_title = "nessai"
html_theme_options = {
    "path_to_docs": "docs",
    "repository_url": "https://github.com/mj-will/nessai",
    "repository_branch": "main",
    "use_edit_page_button": True,
    "use_issues_button": True,
    "use_repository_button": True,
    "use_download_button": True,
}

# -- Configure autoapi -------------------------------------------------------

autoapi_type = "python"
autoapi_dirs = ["../src/nessai/"]
autoapi_add_toctree_entry = False
autoapi_options = ["members", "show-inheritance", "show-module-summary"]

# -- RST config --------------------------------------------------------------
# Inline python highlighting, base on https://stackoverflow.com/a/68931907

rst_prolog = """
.. role:: python(code)
    :language: python
    :class: highlight
"""
