"""Wrapper to expose the FFD function from the TessFlareLightcurves module."""

import os
import sys

# Add the TessFlareLightcurves directory so the vendored FFD module is importable
_sTessDir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'TessFlareLightcurves')
if _sTessDir not in sys.path:
    sys.path.insert(0, _sTessDir)

from FFD import FFD  # noqa: F401, E402
