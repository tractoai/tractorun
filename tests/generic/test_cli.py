from tests.utils import TractoCli
from tractorun import __version__


def test_version_argument() -> None:
    tracto_cli = TractoCli(
        args=[
            "--version",
        ],
    )
    op_run = tracto_cli.run()
    op_run.validate_exit_code()
    assert op_run.stdout.decode().strip() == f"tractorun {__version__}"
