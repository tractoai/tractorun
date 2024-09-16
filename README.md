### Install local environment
1. Install [pyenv](https://github.com/pyenv/pyenv)
2. Create and activate new env `pyenv virtualenv 3.10 tractorun && pyenv activate tractorun`
3. Install all dependencies: `pip install --extra-index-url https://artifactory.nebius.dev/artifactory/api/pypi/nyt/simple ."[all]`


### Build and push new runtime image
```shell
./run_build torchesaurus_runtime --push
```

### Build and push new image for tests
```shell
./run_build torchesaurus_tests --push
./run_build tensorproxy_tests --push
```
and update image in `run_tests`

### Build and push new demo image
```shell
./run_build demo_image --push
```

### Run tests
```shell
YT_PROXY=dirac.yt.nemax.nebiuscloud.net YT_TOKEN=<...> ./run_tests . -s
```

It is possible to run separate test suites or individual tests:
```shell
YT_PROXY=dirac.yt.nemax.nebiuscloud.net YT_TOKEN=<...> ./run_tests general . -s
YT_PROXY=dirac.yt.nemax.nebiuscloud.net YT_TOKEN=<...> ./run_tests tensorproxy . -s
```

To run an individual test:
```shell
YT_PROXY=dirac.yt.nemax.nebiuscloud.net YT_TOKEN=<...> ./run_tests generic test_sidecars.py
YT_PROXY=dirac.yt.nemax.nebiuscloud.net YT_TOKEN=<...> ./run_tests generic test_sidecars.py::test_sidecar_run
```

### Bump library version
Just run `Create release` action.

### Build and upload a wheel
Just run `Build and upload to internal pypi` action.

Or you can do it manually, if you are brave enough:

- log in to https://artifactory.nebius.dev/ui/packages
- go to `Welcome, username` -> `Edit Profile` -> `Generate an Identity Token`
- make your `~/.pypirc` look this way:
```
[distutils]
index-servers = local
[local]
repository: https://artifactory.nebius.dev/artifactory/api/pypi/nyt
username: username@nebius.com
password: <your token>
```

Now you can do it!
```shell
./build_and_upload.sh
```
Make sure your git revision has a version tag.
