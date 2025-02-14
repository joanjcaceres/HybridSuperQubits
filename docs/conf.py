# Configuration file for the Sphinx documentation builder.

# -- Project information -----------------------------------------------------
project = 'HybridSuperQubits'
copyright = '2025, Joan Caceres'
author = 'Joan Caceres'
release = '0.1'

# -- General configuration ---------------------------------------------------
import os
import sys
sys.path.insert(0, os.path.abspath('..'))  # Permite que Sphinx encuentre los módulos

extensions = [
    'sphinx.ext.autodoc',   # Documentar automáticamente desde docstrings
    'sphinx.ext.napoleon',  # Soporte para docstrings estilo Google/NumPy
    'sphinx.ext.viewcode',  # Agrega enlaces al código fuente en la doc
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
html_theme = "sphinx_rtd_theme"  # Tema de Read the Docs
html_static_path = ['_static']