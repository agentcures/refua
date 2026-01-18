import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.abspath("../../src"))

project = "Refua"
author = "Refua"
copyright = f"{datetime.now().year}, {author}"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
    "myst_parser",
]

autosummary_generate = True
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_attr_annotations = True
autodoc_typehints = "description"
autodoc_member_order = "bysource"

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

master_doc = "index"

html_theme = "furo"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}
