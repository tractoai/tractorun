import importlib
import inspect
import pkgutil
from types import ModuleType
from typing import Iterable


def get_all_modules() -> Iterable[ModuleType]:
    import tractorun

    for module_info in pkgutil.walk_packages(tractorun.__path__, tractorun.__name__ + "."):
        yield importlib.import_module(module_info.name)


def is_private_module(module: ModuleType | None) -> bool:
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


def test_imports() -> None:
    invalid_public_modules: list[tuple[str, object, ModuleType]] = []
    for module in get_all_modules():
        if is_private_module(module):
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
            if not is_private_module(obj_module):
                continue
            invalid_public_modules.append((name, obj, module))
    assert invalid_public_modules == []
