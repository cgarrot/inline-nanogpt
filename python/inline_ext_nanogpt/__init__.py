"""Inline Studio / OpenChar extension: NanoGPT hosted models (https://nano-gpt.com).

One ``register(reg)`` for the whole extension. The registrar instantiates runner
classes with no arguments, so per-extension state (the private data dir, used for
the API key fallback file and downloads) is captured here at register time.
"""

from __future__ import annotations

from inline_core.extensions.api import ExtensionRegistrar

from . import api
from .nodes import (
    NODE_CATALOGS,
    NanoGPTAudioNode,
    NanoGPTImageNode,
    NanoGPTTextNode,
    NanoGPTVideoNode,
    apply_catalogs,
)


def register(reg: ExtensionRegistrar) -> None:
    api.DATA_DIR = reg.data_dir
    # Swap the placeholder TEXT model param for a SELECT over the full live catalog
    # (disk-cached 12h, offline fallback; a restart refreshes the list).
    apply_catalogs()
    reg.nodes(NanoGPTImageNode, NanoGPTTextNode, NanoGPTVideoNode, NanoGPTAudioNode)
