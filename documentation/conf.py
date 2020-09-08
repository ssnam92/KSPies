# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import os
import sys

#currentpath = os.path.dirname(os.path.abspath('.'))
#currentpath = os.path.join(currentpath, 'kspies')
#print(currentpath)
#sys.path.insert(0, os.path.dirname(currentpath))


documentation_path = os.path.abspath('.')
top_dir = os.path.dirname(documentation_path)
print(top_dir)
code_path = os.path.join(top_dir, 'kspies')
print(code_path)
#print(os.path.abspath(top_dir))
sys.path.insert(0, code_path)
#print(os.path.abspath("..kspies"))
#sys.path.insert(0, os.path.abspath("..kspies"))



# -- Project information -----------------------------------------------------

project = 'KS Pies'
copyright = '2020, Seungsoo Nam, Ryan J. McCarty, Hansol Park, Eunji Sim'
author = 'Seungsoo Nam, Ryan J. McCarty,  Hansol Park, Eunji Sim'

# The full version, including alpha/beta/rc tags
release = '2020'



# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.autodoc', 'sphinx.ext.napoleon', 'sphinx.ext.todo', 'sphinx.ext.doctest', 'sphinx.ext.viewcode' ]

napoleon_google_docstring=True
napoleon_numpy_docstring=True
napoleon_use_param=True
napoleon_use_ivar=True

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'
#html_theme = 'nature'
#html_theme_options = {body_max_width:0} #
# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
html_logo = 'KSPies.svg'
html_favicon = 'favicon-32x32.png'
html_theme_options = {
    'logo_only': False,
    'display_version': False,
}


autosummary_generate=True

autodoc_mock_imports = ["pyscf", "pyfort"]
todo_include_todos = True
show_authors = True
autodoc_member_order = 'bysource'

pygments_style='sphinx'
