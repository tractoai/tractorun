![img.png](docs/_static/img.png)

# 🚜 Tractorun

`tractorun` is a powerful tool for distributed ML operations on the [Tracto.ai](https://tracto.ai/) platform.

Also, `tractorun` can be used to run arbitrary [gang operations](https://en.wikipedia.org/wiki/Gang_scheduling) on the Tracto.ai platform.

TODO:
1. minimal changes
2. binds, stderr-reader
3. torch and jax
4. example config
5. example script

# Getting started

Install tractorun into your python3 environment:

`pip install --upgrade tractorun`

Configure client to work with your cluster: 

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

# How to run

## CLI

`tractorun --help`

or with yaml config

`tractorun --run-config-path config.yaml`

You can find a relevant example in [this repository](https://github.com/tractoai/tractorun/tree/main/examples/pytorch/lightning_mnist_ddp_script).

## Python SDK

SDK is convenient to use from Jupyter notebooks for development purposes.

You can find a relevant example in [this repository](https://github.com/tractoai/tractorun/tree/main/examples/pytorch/lightning_mnist).

WARNING: your local environment should be equals to remote docker image on TractoAI platform to use SDK.
* This requirement is met in Jupyter Notebook on Tracto.ai platform.
* For local use, it is recommended to run the code locally in the same container as specified in the docker_image parameter in `tractorun`

# How to adapt code for tractorun

## CLI

1. Wrap all training/inference code to a function.
2. Initiate environment and Toolbox by `from tractorun.run.prepare_and_get_toolbox`

An example of adapting the mnist training from the [PyTorch repository](https://github.com/pytorch/examples/blob/cdef4d43fb1a2c6c4349daa5080e4e8731c34569/mnist/mnist_simple/main.py): https://github.com/tractoai/tractorun/tree/main/examples/adoptation/mnist_simple/cli

## SDK

1. Wrap all training/inference code to a function with a `toolbox: tractorun.toolbox.Toolbox` parameter.
2. Run this function by `tractorun.run.run`.

An example of adapting the mnist training from the [PyTorch repository](https://github.com/pytorch/examples/blob/cdef4d43fb1a2c6c4349daa5080e4e8731c34569/mnist/main.py): https://github.com/tractoai/tractorun/tree/main/examples/adoptation/mnist_simple/sdk

# Features

## Toolbox

## Backends

Backends configure `tractorun` to work with a specific ML framework.

Tractorun supports multiple backends:
* [Tractorch](https://github.com/tractoai/tractorun/tree/main/tractorun/backend/tractorch) for PyTorch
  * [examples](https://github.com/tractoai/tractorun/tree/main/examples/pytorch)
* [Tractorax](https://github.com/tractoai/tractorun/tree/main/tractorun/backend/tractorax) for Jax
  * [examples](https://github.com/tractoai/tractorun/tree/main/examples/jax)
* [Generic](https://github.com/tractoai/tractorun/tree/main/tractorun/backend/generic)
  * non-specialized backend, can be used as a basis for other backends

# Options and arguments

[Options reference](https://github.com/tractoai/tractorun/blob/main/docs/options.md) page provides an overview of all available options for `tractorun`, explaining their purpose and usage. Options can be defined by:
* cli parameters
* yaml config
* python options

# Development

## Install local environment
1. Install [pyenv](https://github.com/pyenv/pyenv)
2. Create and activate new env `pyenv virtualenv 3.10 tractorun && pyenv activate tractorun`
3. Install all dependencies: `pip install ."[all]`


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
