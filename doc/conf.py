#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# prospector documentation build configuration file, created by
# sphinx-quickstart on Sun Apr  8 16:26:26 2018.
#
# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('..'))

# -- General configuration ------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",
    "myst_nb",
    "sphinx.ext.napoleon",
    "numpydoc"

]

myst_enable_extensions = ["dollarmath", "colon_fence"]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']
source_suffix = '.rst'

# The encoding of source files.
# source_encoding = 'utf-8-sig'

# The master toctree document.
master_doc = 'index'

# General information about the project.
project = 'prospector'
copyright = '2014-2023, Benjamin Johnson and Contributors'
author = 'Benjamin Johnson'

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
# The short X.Y version.
version = '1.2'
release = '1.2'

language = None
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# If true, '()' will be appended to :func: etc. cross-reference text.
# add_function_parentheses = True
# If true, the current module name will be prepended to all description
# unit titles (such as .. function::).
# add_module_names = True

autodoc_mock_imports = ["sedpy", "h5py"]
# The name of the Pygments (syntax highlighting) style to use.
#pygments_style = 'sphinx'

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = False


# -- Options for HTML output ----------------------------------------------
html_title = "prospector"
htmlhelp_basename = 'prospectordoc'
html_logo = "_static/logo_name_kh.png"
html_favicon = "_static/favicon.png"
html_theme = 'sphinx_book_theme'
html_static_path = ["_static"]
html_css_files = ['css/custom.css']

html_copy_source = True
html_show_sourcelink = True
html_theme_options = {"path_to_docs": "doc",
                      "repository_url": "https://github.com/bd-j/prospector",
                      "repository_branch": "main",
                      "use_repository_button": True,
                      "use_edit_page_button": True,
                      "use_issues_button": True,
                      "use_download_button": True,
                      "logo_only": True,
                      "extra_navbar": ("<p>Logo by Lynn Carlson & Kate Whitaker</p>"
                                       "<p>Theme by Executable Book Project</p>"),}


# If not None, a 'Last updated on:' timestamp is inserted at every page
# bottom, using the given strftime format.
# The empty string is equivalent to '%b %d, %Y'.
html_last_updated_fmt = ''

# Output file base name for HTML help builder.

autodoc_default_options = {
    'member-order': 'bysource',
}

#def skip(app, what, name, obj, would_skip, options):
#    if name == "__init__":
#        return False
#    return would_skip

#def setup(app):
#    app.connect("autodoc-skip-member", skip)

# -- Options for LaTeX output ---------------------------------------------

latex_elements = {
     # The paper size ('letterpaper' or 'a4paper').
     # 'papersize': 'letterpaper',
     # The font size ('10pt', '11pt' or '12pt').
     # 'pointsize': '10pt',
     # Additional stuff for the LaTeX preamble.
     # 'preamble': '',
     # Latex figure (float) alignment
     # 'figure_align': 'htbp',
}
