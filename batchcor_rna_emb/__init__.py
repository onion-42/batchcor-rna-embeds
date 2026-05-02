"""Batch correction and downstream evaluation for scGPT RNA embeddings (Group 7)."""

# Must run before any submodule imports scgpt: torchtext 0.18.0 ships a
# ``libtorchtext.pyd`` that refuses to load on torch >= 2.4 and the package
# is archived upstream, so we substitute a pure-Python stand-in.
from . import _torchtext_shim  # noqa: F401

__version__ = "0.0.1"
