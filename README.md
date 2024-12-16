# Tractorun

`tractorun` is a powerful tool for distributed ML operations on the [Tracto.ai](https://tracto.ai/) platform.

Also, `tractorun` can be used to run arbitrary [gang operations](https://en.wikipedia.org/wiki/Gang_scheduling) on the Tracto.ai platform.

TODO:
1. minimal changes
2. binds, stderr-reader
3. torch and jax
4. example config
5. example script

# Getting start

Install tractorun into your python3 environment:

`pip install --upgrade tractorun`

# How to try

Run example script:

```
tractorun \
    --yt-path "//tmp/$USER/tractorun_getting_started" \
    --bind-local './examples/pytorch/lightning_mnist_ddp_script/lightning_mnist_ddp_script.py:/lightning_mnist_ddp_script.py' \
    --bind-local-lib ./tractorun \
    --docker-image cr.ai.nebius.cloud/crnf2coti090683j5ssi/tractorun/examples_runtime:2024-11-20-20-00-05 \
    python3 /lightning_mnist_ddp_script.py
```

# How to adapt code for tractorun

# Backends

TODO

# Options

[Documentation](https://github.com/tractoai/tractorun/blob/main/docs/options.md)

# Development

## Install local environment
1. Install [pyenv](https://github.com/pyenv/pyenv)
2. Create and activate new env `pyenv virtualenv 3.10 tractorun && pyenv activate tractorun`
3. Install all dependencies: `pip install --extra-index-url https://artifactory.nebius.dev/artifactory/api/pypi/nyt/simple ."[all]`


## Build new image for tests
```shell
./run_build generic
./run_build tractorch_tests
./run_build tractorax_tests
./run_build tensorproxy_tests
```
and update images in `./run_tests` and `tests/utils.py`

## Build and push new image for examples

```shell
./run_build examples_runtime --push
```

and update image in `./examples/run_example`

### Run tests

To run tests on local YT run `pytest`
```shell
./run_tests all . -s
```

To run tests on remote cluster
```shell
./run_tests general . -s
./run_tests tensorproxy . -s
```

It is possible to provide extra `pytest` options
```shell
./run_tests generic test_sidecars.py
./run_tests generic test_sidecars.py::test_sidecar_run
```

## Build and upload
1. Run [Create release](https://github.com/tractoai/tractorun/actions/workflows/release.yaml)
2. Run [Build and upload to external pypi](https://github.com/tractoai/tractorun/actions/workflows/pypi_external.yaml). Specify the latest tag from [the list](https://github.com/tractoai/tractorun/tags) to upload the latest version.
