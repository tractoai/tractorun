import importlib
import pkgutil
from types import ModuleType
from typing import Iterable

import pytest

from tests.utils import check_no_private_objects_in_public_namespace


def _get_all_modules() -> Iterable[ModuleType]:
    import tractorun

    for module_info in pkgutil.walk_packages(tractorun.__path__, tractorun.__name__ + "."):
        if not module_info.name.startswith("tractorun.backend.tractorax"):
            continue
        yield importlib.import_module(module_info.name)


def test_no_private_objects_in_public_namespace(can_test_jax: bool) -> None:
    if not can_test_jax:
        pytest.skip("jax can't be run on this platform")
    assert check_no_private_objects_in_public_namespace(_get_all_modules())
