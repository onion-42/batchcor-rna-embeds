"""
Drop-in stub for the deprecated `torchtext` package.

Background
----------
``scgpt 0.2.4`` imports ``torchtext.vocab.Vocab`` at module load time, but the
last torchtext release (0.18.0) was compiled against torch 2.3 and its native
extension ``libtorchtext.pyd`` refuses to load on torch >= 2.4.  The package is
also archived upstream, so there is no newer wheel.

This module is imported as the very first thing inside :mod:`batchcor_rna_emb`
so it lands in ``sys.modules`` *before* any scgpt code runs.  scgpt's
``GeneVocab`` then transparently inherits from our pure-Python ``_MiniVocab``
implementation, which is sufficient for vocabulary-only operations
(``__len__``, ``__getitem__``, ``get_stoi``, ``set_default_index`` …).

The shim is a no-op when a real, working ``torchtext`` is already importable.
"""
from __future__ import annotations

import sys
import types
from collections import OrderedDict
from typing import Iterable

# ---------------------------------------------------------------------------
# Skip if a real torchtext loads successfully.
# ---------------------------------------------------------------------------
_REAL_TORCHTEXT_OK = False
if "torchtext" in sys.modules:
    _REAL_TORCHTEXT_OK = True
else:
    try:
        import torchtext as _real_torchtext  # noqa: F401

        _REAL_TORCHTEXT_OK = True
    except (ImportError, OSError):
        # Either not installed (ImportError) or .pyd refuses to load
        # against the current torch (OSError on Windows).
        for _mod in [
            m for m in list(sys.modules) if m == "torchtext" or m.startswith("torchtext.")
        ]:
            sys.modules.pop(_mod, None)


# ---------------------------------------------------------------------------
# Minimal Vocab implementation (only what scgpt 0.2.x exercises).
# ---------------------------------------------------------------------------
class _MiniVocab:
    """Pure-Python token<->index mapping mimicking the slice of
    ``torchtext.vocab.Vocab`` that scgpt actually relies on."""

    def __init__(self, source: "dict | _MiniVocab | list") -> None:
        if isinstance(source, _MiniVocab):
            token2idx = dict(source._token2idx)
        elif isinstance(source, dict):
            token2idx = dict(source)
        elif isinstance(source, list):
            token2idx = {tok: i for i, tok in enumerate(source)}
        else:
            raise TypeError(
                f"_MiniVocab cannot be built from {type(source).__name__}"
            )
        self._token2idx: dict[str, int] = token2idx
        max_idx = max(self._token2idx.values()) if self._token2idx else -1
        self._idx2token: list[str | None] = [None] * (max_idx + 1)
        for tok, idx in self._token2idx.items():
            self._idx2token[idx] = tok
        self._default_idx: int = -1

    # Sequence protocol -----------------------------------------------------
    def __len__(self) -> int:
        return len(self._token2idx)

    def __getitem__(self, key):
        if isinstance(key, str):
            if key in self._token2idx:
                return self._token2idx[key]
            if self._default_idx >= 0:
                return self._default_idx
            raise KeyError(key)
        if isinstance(key, int):
            return self._idx2token[key]
        raise TypeError(f"_MiniVocab does not support key type {type(key).__name__}")

    def __contains__(self, key) -> bool:
        return key in self._token2idx

    def __iter__(self):
        return iter(self._token2idx)

    # Public API used by scgpt ---------------------------------------------
    def get_stoi(self) -> dict[str, int]:
        return dict(self._token2idx)

    def get_itos(self) -> list[str]:
        return [t for t in self._idx2token if t is not None]

    def set_default_index(self, idx: int) -> None:
        self._default_idx = int(idx)

    def lookup_indices(self, tokens: Iterable[str]) -> list[int]:
        return [self.__getitem__(t) for t in tokens]

    def lookup_tokens(self, indices: Iterable[int]) -> list[str]:
        return [self._idx2token[i] for i in indices]

    # ``GeneVocab.__init__`` does ``super().__init__(_vocab.vocab)``.
    # Returning self lets that path work transparently.
    @property
    def vocab(self) -> "_MiniVocab":
        return self

    def append_token(self, token: str) -> None:
        if token in self._token2idx:
            return
        new_idx = len(self._idx2token)
        self._token2idx[token] = new_idx
        self._idx2token.append(token)

    def insert_token(self, token: str, index: int) -> None:
        """Insert ``token`` at position ``index`` (amortized O(1) when the
        target index lands at the end of the vocabulary, which is the
        access pattern used by ``GeneVocab.from_dict``)."""
        if token in self._token2idx:
            return
        n = len(self._idx2token)
        if index < 0 or index > n:
            raise IndexError(
                f"insert_token index {index} out of range [0, {n}]"
            )
        if index == n:
            self._idx2token.append(token)
            self._token2idx[token] = index
            return
        # Generic shift path -- O(n).  Only triggered if scgpt deviates
        # from the sorted-index `from_dict` pattern.
        self._idx2token.insert(index, token)
        for i in range(index, len(self._idx2token)):
            tok = self._idx2token[i]
            if tok is not None:
                self._token2idx[tok] = i


def _vocab_from_ordered_dict(
    ordered_dict: "dict[str, int]", min_freq: int = 1
) -> _MiniVocab:
    """Stand-in for ``torchtext.vocab.vocab(ordered_dict, min_freq=...)``."""
    keep = [tok for tok, freq in ordered_dict.items() if freq >= min_freq]
    return _MiniVocab(keep)


def _build_vocab_from_iterator(
    iterator: Iterable[Iterable[str] | str],
    min_freq: int = 1,
    specials: list[str] | None = None,
    special_first: bool = True,
) -> _MiniVocab:
    """Light-weight stand-in for ``torchtext.vocab.build_vocab_from_iterator``."""
    if specials is None:
        specials = []
    counter: dict[str, int] = {}
    for item in iterator:
        if isinstance(item, str):
            counter[item] = counter.get(item, 0) + 1
        else:
            for tok in item:
                counter[tok] = counter.get(tok, 0) + 1
    ordered = [t for t, c in counter.items() if c >= min_freq]
    final: list[str] = []
    if special_first:
        final.extend(specials)
    for tok in ordered:
        if tok not in specials:
            final.append(tok)
    if not special_first:
        final.extend(specials)
    return _MiniVocab(final)


# ---------------------------------------------------------------------------
# Inject the fake module hierarchy when the real one is unusable.
# ---------------------------------------------------------------------------
if not _REAL_TORCHTEXT_OK:
    _torchtext_mod = types.ModuleType("torchtext")
    _vocab_mod = types.ModuleType("torchtext.vocab")
    _vocab_mod.Vocab = _MiniVocab
    _vocab_mod.vocab = _vocab_from_ordered_dict
    _vocab_mod.build_vocab_from_iterator = _build_vocab_from_iterator
    _torchtext_mod.vocab = _vocab_mod
    _torchtext_mod.__version__ = "0.0.0+stub"

    # Empty placeholders that scgpt sometimes touches lazily
    _torchtext_mod.__path__ = []
    _vocab_mod.__path__ = []

    sys.modules["torchtext"] = _torchtext_mod
    sys.modules["torchtext.vocab"] = _vocab_mod
