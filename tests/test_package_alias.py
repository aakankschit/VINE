"""Tests for the deprecated ``maple_fep`` alias of ``vine_fep``."""

import importlib
import sys
import warnings


def _fresh_import_maple_fep():
    for name in [
        m for m in sys.modules if m == "maple_fep" or m.startswith("maple_fep.")
    ]:
        del sys.modules[name]
    return importlib.import_module("maple_fep")


def test_import_warns_and_aliases_vine_fep():
    import vine_fep

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        maple_fep = _fresh_import_maple_fep()
    assert maple_fep is vine_fep
    assert any(issubclass(x.category, DeprecationWarning) for x in w)


def test_submodule_imports_resolve_to_same_objects():
    import vine_fep.models
    import vine_fep.models.probabilistic.mixture.presets as new_presets

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        _fresh_import_maple_fep()
        import maple_fep.models.probabilistic.mixture.presets as old_presets
        from maple_fep.models import MixtureVI

    assert MixtureVI is vine_fep.models.MixtureVI
    assert old_presets is new_presets
    assert new_presets.__spec__.name == "vine_fep.models.probabilistic.mixture.presets"
