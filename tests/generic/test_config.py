import pytest

from tests.utils import run_config_file
from tractorun.cli.tractorun_runner import make_configuration
from tractorun.exception import TractorunConfigError


def test_fail_unknown_key() -> None:
    with run_config_file({"__SOME_UNKNOWN_KEY__": []}) as run_config_path:
        with pytest.raises(TractorunConfigError):
            _, _, config = make_configuration(["--run-config-path", run_config_path])
