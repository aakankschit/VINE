"""
Deprecated alias for :mod:`vine_fep`.

The package was renamed from ``maple-fep`` to ``vine-fep`` in 0.2.0.
``import maple_fep`` (and any ``maple_fep.<submodule>`` import) still works
and returns the same module objects as ``vine_fep``, but emits a
``DeprecationWarning``. This alias will be removed in 0.3.0.
"""

import importlib
import importlib.abc
import importlib.util
import sys
import warnings

import vine_fep

warnings.warn(
    "maple_fep has been renamed to vine_fep; `import maple_fep` will stop "
    "working in 0.3.0. Use `import vine_fep` instead.",
    DeprecationWarning,
    stacklevel=2,
)

_OLD = "maple_fep"
_NEW = "vine_fep"


class _AliasFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Resolve ``maple_fep.<x>`` imports to the already-importable ``vine_fep.<x>``."""

    def find_spec(self, fullname, path, target=None):  # noqa: ARG002
        if not fullname.startswith(_OLD + "."):
            return None
        new_name = _NEW + fullname[len(_OLD) :]
        if importlib.util.find_spec(new_name) is None:
            return None
        return importlib.util.spec_from_loader(fullname, self)

    def create_module(self, spec):
        module = importlib.import_module(_NEW + spec.name[len(_OLD) :])
        spec.loader_state = module.__spec__
        return module

    def exec_module(self, module):
        # The import system overwrote __spec__ with the alias spec; restore it.
        module.__spec__ = module.__spec__.loader_state


if not any(isinstance(f, _AliasFinder) for f in sys.meta_path):
    sys.meta_path.insert(0, _AliasFinder())

sys.modules[__name__] = vine_fep
