"""Availability probe for the optional native UVA gather extension.

Mirrors :mod:`uni_rl.ipc.replay_pipelines.native_h2d`: build prerequisites are
pre-validated so callers can report why the UVA path is unavailable, and every
failure falls back to the pinned + side-stream H2D path.
"""

from __future__ import annotations

import logging
import shutil
from functools import lru_cache
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_SOURCE = Path(__file__).with_name("layered_uva_ext.cu")
_NATIVE_AVAILABLE: bool | None = None
_DIAGNOSTIC: str = ""


def _check_build_prerequisites() -> tuple[bool, str]:
    """Pre-validate JIT compilation requirements."""
    try:
        from torch.utils.cpp_extension import is_ninja_available

        if not is_ninja_available():
            return False, "ninja not found. Install: pip install ninja"
    except ImportError:
        return False, "torch.utils.cpp_extension unavailable"

    from torch.utils.cpp_extension import get_cxx_compiler

    compiler = get_cxx_compiler()
    if shutil.which(compiler) is None:
        return False, (f"C++ compiler '{compiler}' not found. Install: sudo apt install g++")

    from torch.utils.cpp_extension import CUDA_HOME

    if CUDA_HOME is None:
        return False, (
            "CUDA_HOME not set and nvcc not found. Install CUDA toolkit or set CUDA_HOME."
        )

    header = Path(CUDA_HOME) / "include" / "cuda_runtime_api.h"
    if not header.exists():
        targets_header = (
            Path(CUDA_HOME) / "targets" / "x86_64-linux" / "include" / "cuda_runtime_api.h"
        )
        if not targets_header.exists():
            return False, (
                f"cuda_runtime_api.h not found in {CUDA_HOME}/include/ "
                f"or targets/x86_64-linux/include/. "
                f"Verify CUDA toolkit installation."
            )

    if not _SOURCE.exists():
        return False, f"Extension source not found: {_SOURCE}"

    return True, ""


@lru_cache(maxsize=1)
def _try_load_extension() -> Any | None:
    global _NATIVE_AVAILABLE, _DIAGNOSTIC

    ok, reason = _check_build_prerequisites()
    if not ok:
        _DIAGNOSTIC = reason
        _NATIVE_AVAILABLE = False
        logger.info("Native UVA gather extension unavailable: %s. Using H2D fallback.", reason)
        return None

    try:
        from torch.utils.cpp_extension import CUDA_HOME, load

        extra_include_paths = []
        if CUDA_HOME is not None:
            extra_include_paths.append(str(Path(CUDA_HOME) / "include"))
            targets_include = Path(CUDA_HOME) / "targets" / "x86_64-linux" / "include"
            if targets_include.exists():
                extra_include_paths.append(str(targets_include))
        ext = load(
            name="unilab_layered_uva",
            sources=[str(_SOURCE)],
            extra_cflags=["-O3"],
            extra_include_paths=extra_include_paths,
            verbose=False,
        )
        _NATIVE_AVAILABLE = True
        return ext
    except Exception as exc:
        _DIAGNOSTIC = f"Compilation failed: {exc}"
        _NATIVE_AVAILABLE = False
        logger.warning("Native UVA gather compilation failed: %s. Using H2D fallback.", exc)
        return None


def is_available() -> bool:
    """True if the native UVA gather extension is loaded or loadable."""
    global _NATIVE_AVAILABLE
    if _NATIVE_AVAILABLE is None:
        _try_load_extension()
    return bool(_NATIVE_AVAILABLE)


def get_diagnostic() -> str:
    """Human-readable reason for unavailability."""
    if _NATIVE_AVAILABLE is None:
        _try_load_extension()
    return _DIAGNOSTIC


def load_extension() -> Any | None:
    """Return the loaded extension module, or None when unavailable."""
    if not is_available():
        return None
    return _try_load_extension()
