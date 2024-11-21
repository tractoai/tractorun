import importlib
import pkgutil
from types import ModuleType
from typing import Iterable

from tests.utils import check_no_private_objects_in_public_namespace


def _get_all_modules() -> Iterable[ModuleType]:
    import tractorun

    for module_info in pkgutil.walk_packages(tractorun.__path__, tractorun.__name__ + "."):
        # should base tractorun logic and  generic backend
        if module_info.name.startswith("tractorun.backend") and not module_info.name.startswith(
            "tractorun.backend.generic"
        ):
            continue
        yield importlib.import_module(module_info.name)


def test_no_private_objects_in_public_namespace() -> None:
    assert check_no_private_objects_in_public_namespace(_get_all_modules())
