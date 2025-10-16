"""
workhealthlab Utils Module
=========================

Utility functions, styling tools, and project templates.

Modules
-------
style
    Color palettes, plot themes, and style utilities.

PowerShell Scripts (in this directory)
---------------------------------------
workhealthlab-init.ps1
    Main utility for creating projects and notebooks.
create-project.ps1
    Generate complete research project structures.
create-notebook.ps1
    Create template Jupyter notebooks for different analysis types.

See utils/README.md for detailed usage instructions.
"""

from . import style

__all__ = [
    'style',
]
