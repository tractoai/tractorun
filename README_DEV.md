# Development

## Install local environment
1. Install [pyenv](https://github.com/pyenv/pyenv)
2. Create and activate new env `pyenv virtualenv 3.10 tractorun && pyenv activate tractorun`
3. Install all dependencies: `pip install ."[all]`


## Build new image for tests
```shell
./run_build.sh generic
./run_build.sh tractorch_tests
./run_build.sh tractorax_tests
./run_build.sh tensorproxy_tests
```
and update images in `./run_tests` and `tests/utils.py`

## Build and push a new image for examples

```shell
./run_build.sh examples_runtime --push
```

and update the image in `./examples/run_example`

## Update current image tag for tests and examples

```shell
./run_update_tag.sh new_tag
```

## Run tests

To run tests on local YT run `pytest`
```shell
./run_tests.sh all . -s
```

To run tests on remote cluster
```shell
./run_tests.sh general . -s
./run_tests.sh tensorproxy . -s
```

It is possible to provide extra `pytest` options
```shell
./run_tests.sh generic test_sidecars.py
./run_tests.sh generic test_sidecars.py::test_sidecar_run
```

## Build and upload
1. Run [Create release](https://github.com/tractoai/tractorun/actions/workflows/release.yaml)
2. Run [Build and upload to external pypi](https://github.com/tractoai/tractorun/actions/workflows/pypi_external.yaml). Specify the latest tag from [the list](https://github.com/tractoai/tractorun/tags) to upload the latest version.
