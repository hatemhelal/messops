from __future__ import annotations

import importlib.metadata

import messops as m


def test_version():
    assert importlib.metadata.version("messops") == m.__version__
