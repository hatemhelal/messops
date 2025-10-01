"""
Copyright (c) 2025 Hatem Helal. All rights reserved.

messops: MESSOPS: Ops for MESS
"""

from __future__ import annotations

import os
import sysconfig
from pathlib import Path

from messops._core import IntegralContext
from messops._version import version as __version__

__all__ = ["IntegralContext", "__version__"]


def _set_libint_data_path() -> None:
    # Get the actual runtime platlib path
    sitepkg = Path(sysconfig.get_path("platlib"))
    data_path = sitepkg / "share" / "libint" / "2.11.1" / "basis"
    os.environ["LIBINT_DATA_PATH"] = str(data_path)

    if not data_path.exists():
        msg = f"Libint data directory not found at {data_path}. "
        msg += "Please ensure the package was installed correctly."
        raise RuntimeError(msg)


_set_libint_data_path()
