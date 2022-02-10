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
copyright = '2014-2022, Benjamin Johnson and Contributors'
author = 'Benjamin Johnson'

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
# The short X.Y version.
version = '1.0'
# The full version, including alpha/beta/rc tags.
release = '1.0'

language = None

# There are two options for replacing |today|: either, you set today to some
# non-false value, then it is used:
# today = ''
#
# Else, today_fmt is used as the format for a strftime call.
# today_fmt = '%B %d, %Y'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This patterns also effect to html_static_path and html_extra_path
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# If true, '()' will be appended to :func: etc. cross-reference text.
# add_function_parentheses = True

# If true, the current module name will be prepended to all description
# unit titles (such as .. function::).
# add_module_names = True

autodoc_mock_imports = ["sedpy", "h5py"]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'


# If true, keep warnings as "system message" paragraphs in the built documents.
# keep_warnings = False

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = False


# -- Options for HTML output ----------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = 'sphinx_book_theme'
html_title = "prospector"
html_copy_source = True
html_show_sourcelink = True
html_theme_options = {"path_to_docs": "docs",
                      "repository_url": "https://github.com/bd-j/prospector",
                      "repository_branch": "main",
                      "use_repository_button": True,
                      "use_edit_page_button": True,
                      "use_issues_button": True,
                      "use_download_button": True,
                      "logo_only": True,
                      "extra_navbar": ("<p>Logo by Lynn Carlson & Kate Whitaker</p>"
                                       "<p>Theme by Executable Book Project</p>"),}


html_static_path = ["_static"]
html_css_files = ['css/custom.css']

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
html_logo = "_static/logo_name_kh.png"

# The name of an image file (relative to this directory) to use as a favicon of
# the docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.
html_favicon = "_static/favicon.png"

# If not None, a 'Last updated on:' timestamp is inserted at every page
# bottom, using the given strftime format.
# The empty string is equivalent to '%b %d, %Y'.
html_last_updated_fmt = ''

# Output file base name for HTML help builder.
htmlhelp_basename = 'prospectordoc'


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

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
#latex_documents = [
#    (master_doc, 'prospector.tex', 'prospector Documentation',
#     'Benjamin Johnson', 'manual'),]
