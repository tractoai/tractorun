import importlib
import inspect
import pkgutil
from types import ModuleType
from typing import Iterable


MODULE_BLACKLIST = {
    # We have to ignore jax-related modules here
    # because jaxlib requires avx support, but orbstack doesn't have it.
    # Maybe we should build jaxlib from source for local test container.
    "tractorun.backend.tractorax",
    "tractorun.backend.tractorax.environment",
    "tractorun.backend.tractorax.backend",
}


def test_no_private_objects_in_public_namespace() -> None:
    # this test can't check variables with primitive types like strings, int, float and so on
    invalid_public_modules: list[tuple[str, object, ModuleType]] = []
    for module in _get_all_modules():
        if _is_private_module(module):
            continue
        for name, obj in inspect.getmembers(module):
            if name.startswith("_"):
                continue
            if not isinstance(obj, object):
                continue
            if inspect.ismodule(obj):
                continue
            obj_module = inspect.getmodule(obj)
            if obj_module is None:
                continue
            if not _is_private_module(obj_module):
                continue
            invalid_public_modules.append((name, obj, module))
    assert invalid_public_modules == []


def _get_all_modules() -> Iterable[ModuleType]:
    import tractorun

    for module_info in pkgutil.walk_packages(tractorun.__path__, tractorun.__name__ + "."):
        if module_info.name in MODULE_BLACKLIST:
            continue
        print(module_info.name, module_info.name in MODULE_BLACKLIST)
        yield importlib.import_module(module_info.name)


def _is_private_module(module: ModuleType | None) -> bool:
    if module is None:
        return False
    private_prefixes = [
        "tractorun.private",
        "tractorun.tests",
        "tractorun.cli",
    ]
    for prefix in private_prefixes:
        if module.__name__.startswith(prefix):
            return True
    return False
