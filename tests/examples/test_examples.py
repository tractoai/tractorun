import json
from pathlib import Path
import subprocess
import sys
from typing import Iterable

import pytest

from tests.utils import (
    EXAMPLES_DOCKER_IMAGE,
    TractoCli,
    get_data_path,
)
from tests.yt_instances import YtInstance


def get_examples_pickle_path(example_type: str) -> Iterable[Path]:
    base_examples_path = Path(get_data_path("../../examples")) / example_type
    for path in base_examples_path.rglob("*.py"):
        assert isinstance(path, Path)
        # filter all examples for tractorun cli
        if (path.parent / "run.sh").exists():
            continue
        yield path


def get_examples_script_path(example_type: str) -> Iterable[Path]:
    base_examples_path = Path(get_data_path("../../examples")) / example_type
    for path in base_examples_path.rglob("run.sh"):
        assert isinstance(path, Path)
        script_path = path.parent / (path.parent.name + ".py")
        yield script_path


def run_example_pickle(path: Path, yt_path: str, dataset_path: str) -> None:
    process = subprocess.Popen(
        [
            "python3",
            path,
            "--yt-home-dir",
            yt_path,
            "--dataset-path",
            dataset_path,
            "--docker-image",
            EXAMPLES_DOCKER_IMAGE,
            # run without gpu
            "--pool-tree",
            "default",
            "--gpu-per-process",
            "0",
        ],
        stdout=sys.stdout,
        stderr=sys.stderr,
    )
    process.wait()
    assert process.returncode == 0


def run_example_script(path: Path, yt_path: str, dataset_path: str) -> None:
    tracto_cli = TractoCli(
        command=["python3", get_data_path(path)],
        docker_image=EXAMPLES_DOCKER_IMAGE,
        args=[
            "--yt-path",
            yt_path,
            "--bind-local",
            f"{get_data_path(str(path))}:{get_data_path(str(path))}",
            "--user-config",
            json.dumps({"dataset_path": dataset_path}),
        ],
    )
    op_run = tracto_cli.run()
    assert op_run.is_exitcode_valid()


@pytest.mark.parametrize("example_path", get_examples_pickle_path(example_type="pytorch"))
def test_pytorch_pickle_examples(yt_instance: YtInstance, yt_path: str, mnist_ds_path: str, example_path: Path) -> None:
    run_example_pickle(
        path=example_path,
        yt_path=yt_path,
        dataset_path=mnist_ds_path,
    )


@pytest.mark.parametrize("example_path", get_examples_pickle_path(example_type="jax"))
def test_jax_pickle_examples(
    can_test_jax: bool, yt_instance: YtInstance, yt_path: str, mnist_ds_path: str, example_path: Path
) -> None:
    if example_path.name == "jax_simple_distributed.py":
        pytest.skip("jax_simple_distributed.py can be run only on GPU backend")
    if not can_test_jax:
        pytest.skip("jax can't be run on this platform")
    run_example_pickle(
        path=example_path,
        yt_path=yt_path,
        dataset_path=mnist_ds_path,
    )


@pytest.mark.parametrize("example_path", get_examples_script_path(example_type="pytorch"))
def test_pytorch_script_examples(yt_instance: YtInstance, yt_path: str, mnist_ds_path: str, example_path: Path) -> None:
    run_example_script(
        path=example_path,
        yt_path=yt_path,
        dataset_path=mnist_ds_path,
    )


@pytest.mark.parametrize("example_path", get_examples_script_path(example_type="jax"))
def test_jax_script_examples(
    can_test_jax: bool, yt_instance: YtInstance, yt_path: str, mnist_ds_path: str, example_path: Path
) -> None:
    if not can_test_jax:
        pytest.skip("jax can't be run on this platform")
    run_example_script(
        path=example_path,
        yt_path=yt_path,
        dataset_path=mnist_ds_path,
    )
