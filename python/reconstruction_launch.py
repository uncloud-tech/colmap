# SPDX-License-Identifier: BSD-3-Clause
"""Prefix-relative entry; called by installed shell launcher with python -I -B."""
from pathlib import Path
import sys

if sys.version_info < (3, 10):
    raise SystemExit('colmap-reconstruct requires Python 3.10 or newer')

sys.path.insert(0, str(Path(__file__).resolve().parent))
from colmap_reconstruction.cli import main

raise SystemExit(main())
