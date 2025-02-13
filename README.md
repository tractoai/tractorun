![img.png](https://raw.githubusercontent.com/tractoai/tractorun/refs/heads/main/docs/_static/img.png)

# 🚜 Tractorun

`Tractorun` is a powerful tool for distributed ML operations on the [Tracto.ai](https://tracto.ai/) platform. It helps manage and run workflows across multiple nodes with minimal changes in the user's code:
* Training and fine-tuning models. Use Tractorun to train models across multiple compute nodes efficiently.
* Offline batch inference. Perform fast and scalable model inference.
* Running arbitrary GPU operations, ideal for any computational tasks that require distributed GPU resources.

## How it works

Tractorun is built on top of [Tracto.ai](https://tracto.ai/) and is responsible for coordinating the execution of distributed machine learning tasks, and it has out-of-the-box integrations with PyTorch and Jax, also it can be easily used for any other training or inference framework.

Key advantages:
* No need to manage your cloud infrastructure, such as configuring Kubernetes cluster, or managing GPU and Infiniband drivers. Tracto.ai  solves all these infrastructure problems for you.
* No need to coordinate distributed processes. Tractorun handles it based on the training configuration: the number of nodes and GPUs used.

Key features:
* Simple distributed task setup, just specify the number of nodes and GPUs.
* Convenient ways to run and configure: CLI, YAML config, and Python SDK.
* Integration with the Tracto.ai platform: use datasets and checkpoints stored in the Tracto.ai storage, build pipelines with Tractorun, MapReduce, Clickhouse, Spark, and more.

# Getting started

To use these examples, you'll need a Tracto account. If you don't have one yet, please sign up at [tracto.ai](https://tracto.ai/).

Install tractorun into your python3 environment:

`pip install --upgrade tractorun`

Configure the client to work with your cluster:
```shell
mkdir ~/.yt
cat <<EOF > ~/.yt
"proxy"={
  "url"="$YT_PROXY";
};
"token"="$YT_TOKEN";
EOF
```

Please put your actual Tracto.ai cluster address to `$YT_PROXY` and your token to `$YT_TOKEN`.

# How to try

Run an example script:

```
tractorun \
    --yt-path "//tmp/$USER/tractorun_getting_started" \
    --bind-local './examples/pytorch/lightning_mnist_ddp_script/lightning_mnist_ddp_script.py:/lightning_mnist_ddp_script.py' \
    --bind-local-lib ./tractorun \
    --docker-image ghcr.io/tractoai/tractorun-examples-runtime:2025-02-10-16-14-27 \
    python3 /lightning_mnist_ddp_script.py
```

# How to run

## CLI

`tractorun --help`

or with yaml config

`tractorun --run-config-path config.yaml`

You can find a relevant examples:
* CLI arguments [example](https://github.com/tractoai/tractorun/tree/main/examples/pytorch/lightning_mnist_ddp_script).
* YAML config [example](https://github.com/tractoai/tractorun/tree/main/examples/pytorch/lightning_mnist_ddp_script_config).

## Python SDK

SDK is convenient to use from Jupyter notebooks for development purposes.

You can find a relevant example in [the repository](https://github.com/tractoai/tractorun/tree/main/examples/pytorch/lightning_mnist).

WARNING: the local environment should be equal to the remote docker image on the TractoAI platform to use SDK.
* This requirement is met in Jupyter Notebook on the Tracto.ai platform.
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

`tractorun.toolbox.Toolbox` provides extra integrations with the Tracto.ai platform:
* Preconfigured client by `toolbox.yt_client`
* Basic checkpoints by `toolbox.checkpoint_manager`
* Control over the operation description in the UI by `toolbox.description_manager`
* Access to coordination information by `toolbox.coordinator`

[Toolbox page](https://github.com/tractoai/tractorun/blob/main/docs/toolbox.md) provides an overview of all available toolbox components.

## Backends

Backends configure `tractorun` to work with a specific ML framework.

Tractorun supports multiple backends:
* [Tractorch](https://github.com/tractoai/tractorun/tree/main/tractorun/backend/tractorch) for PyTorch
  * [examples](https://github.com/tractoai/tractorun/tree/main/examples/pytorch)
* [Tractorax](https://github.com/tractoai/tractorun/tree/main/tractorun/backend/tractorax) for Jax
  * [examples](https://github.com/tractoai/tractorun/tree/main/examples/jax)
* [Generic](https://github.com/tractoai/tractorun/tree/main/tractorun/backend/generic)
  * non-specialized backend, can be used as a basis for other backends

# Options and settings

[Options reference](https://github.com/tractoai/tractorun/blob/main/docs/options.md) page provides an overview of all available options for `tractorun`, explaining their purpose and usage. Options can be defined by:
* CLI parameters
* yaml config
* python options

# More information

* [Examples](https://github.com/tractoai/tractorun/tree/main/examples)
* [More examples in Jupyter Notebooks](https://github.com/tractoai/tracto-examples)

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
