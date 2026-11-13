"""Sphinx configuration of the BBarolo documentation (the website: bbarolo.readthedocs.io).

Build locally from the repository folder:

    pip install -r docs/requirements.txt
    sphinx-build -W --keep-going -n -b html docs docs/_build/html
"""

import datetime
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))          # the bbarolo package of this repository
sys.path.insert(0, os.path.join(HERE, "_ext"))     # local extensions (bbparams)

from bbarolo._version import __version__           # noqa: E402

# -- Project -----------------------------------------------------------------
project = "BBarolo"
author = "Enrico Di Teodoro"
copyright = f"2015–{datetime.date.today().year}, Enrico Di Teodoro"
version = release = __version__

# -- General -----------------------------------------------------------------
extensions = [
    "myst_parser",
    "sphinx_design",
    "sphinx_copybutton",
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "bbparams",
]
source_suffix = {".md": "markdown", ".rst": "restructuredtext"}
root_doc = "index"
exclude_patterns = ["_build", "_legacy", "_redirects", "Thumbs.db", ".DS_Store"]
myst_enable_extensions = ["colon_fence", "dollarmath", "amsmath", "deflist", "attrs_inline", "substitution"]
myst_heading_anchors = 3
myst_substitutions = {"version": version}
copybutton_prompt_text = r"\$ |>>> |\.\.\. "
copybutton_prompt_is_regexp = True
autodoc_member_order = "bysource"
autodoc_typehints = "none"
pygments_style = "default"
pygments_dark_style = "github-dark"

# -- HTML ----------------------------------------------------------------------
html_theme = "pydata_sphinx_theme"
html_title = f"BBarolo {version}"
html_static_path = ["_static"]
# The archived documentation of the C++ BBarolo 1.8, served as it is at legacy/1.8/ (do not edit those HTML
# files). Built once from the branch docs of github.com/editeodoro/Bbarolo (commit in legacy/1.8/BUILD.txt) with,
# for the build only: ngc2403.fits left out (the pages link to the copy on GitHub), sphinx.ext.mathjax instead
# of imgmath, no copies of the sources, a template _templates/layout.html that adds the banner
# "BBarolo 1.8 (C++), archived" to every page, and the pyBBarolo of that branch on PYTHONPATH (API page).
html_extra_path = ["_legacy", "_redirects"]
# _redirects: the old pages of the C++ documentation (bbarolo.readthedocs.io before BBarolo 2) redirect to their
# new places, so that links to them keep working.
html_css_files = ["custom.css"]
html_show_sourcelink = False
html_theme_options = {
    "logo": {"text": "3D-Barolo"},
    "github_url": "https://github.com/editeodoro/bbarolo2",
    "navbar_align": "left",
    "header_links_before_dropdown": 6,
    "show_toc_level": 2,
    "navigation_with_keys": False,
    "secondary_sidebar_items": ["page-toc"],
    "footer_start": ["copyright"],
    "footer_end": ["sphinx-version"],
}
html_context = {"default_mode": "auto"}

# -- LaTeX (the PDF of Read the Docs) -------------------------------------------
latex_documents = [(root_doc, "bbarolo.tex", "BBarolo documentation", author, "manual")]


# -- The docstrings of the code are plain text (not reStructuredText): shown as they are ------------------
def _plain_docstring(app, what, name, obj, options, lines):
    if lines:
        lines[:] = ["::", ""] + ["    " + ln for ln in lines] + [""]


# -- The parameter pages of the tasks, generated from the keyword table at every build -----------------
# (folder of the task in tasks/, task name of the keywords, title); folder None: tasks/general.md
PARAMETER_PAGES = [
    (None, "ALL", "General parameters"),
    ("3dfit", "3DFIT", "3DFIT"), ("bayesian3dfit", "BAYESIAN3DFIT", "BAYESIAN3DFIT"),
    ("bayesianslitfit", "BAYESIANSLITFIT", "BAYESIANSLITFIT"), ("galmod", "GALMOD", "GALMOD"),
    ("galwind", "GALWIND", "GALWIND"), ("2dfit", "2DFIT", "2DFIT"), ("spacepar", "SPACEPAR", "SPACEPAR"),
    ("search", "SEARCH", "SEARCH"), ("makemask", "MAKEMASK", "MAKEMASK"), ("smooth", "SMOOTH", "SMOOTH"),
    ("smoothspec", "SMOOTHSPEC", "SMOOTHSPEC"), ("contsub", "CONTSUB", "CONTSUB"), ("stats", "STATS", "STATS"),
    ("maps", "MAPS", "Moment maps and profiles"), ("pvslice", "PVSLICE", "PVSLICE"), ("ellprof", "ELLPROF", "ELLPROF"),
    ("rend3d", "REND3D", "REND3D"),
]


def _parameter_pages(app):
    from bbparams import write_parameter_pages
    write_parameter_pages(app.srcdir, PARAMETER_PAGES)


def setup(app):
    app.connect("autodoc-process-docstring", _plain_docstring)
    app.connect("builder-inited", _parameter_pages)
