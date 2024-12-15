## Development

### Install local environment
1. Install [pyenv](https://github.com/pyenv/pyenv)
2. Create and activate new env `pyenv virtualenv 3.10 tractorun && pyenv activate tractorun`
3. Install all dependencies: `pip install --extra-index-url https://artifactory.nebius.dev/artifactory/api/pypi/nyt/simple ."[all]`


### Build new image for tests
```shell
./run_build generic
./run_build tractorch_tests
./run_build tractorax_tests
./run_build tensorproxy_tests
```
and update images in `./run_tests` and `tests/utils.py`

### Build and push new image for examples

```shell
./run_build examples_runtime --push
```

and update image in `./examples/run_example`

### Run tests

To run tests on local YT run `pytest`
```shell
YT_PROXY=<proxy url> YT_TOKEN=<token> ./run_tests all . -s
```

To run tests on remote cluster
```shell
YT_PROXY=<proxy url> YT_TOKEN=<...> ./run_tests general . -s
YT_PROXY=<proxy url> ./run_tests tensorproxy . -s
```

It is possible to provide extra `pytest` options
```shell
YT_PROXY=dirac.yt.nebius.yt YT_TOKEN=<...> ./run_tests generic test_sidecars.py
YT_PROXY=dirac.yt.nebius.yt YT_TOKEN=<...> ./run_tests generic test_sidecars.py::test_sidecar_run
```

### Build and upload
1. Run [Create release](https://github.com/tractoai/tractorun/actions/workflows/release.yaml)
2. Run [Build and upload to external pypi](https://github.com/tractoai/tractorun/actions/workflows/pypi_external.yaml). Specify the latest tag from [the list](https://github.com/tractoai/tractorun/tags) to upload the latest version.

### Bump library version
Just run `Create release` action.
