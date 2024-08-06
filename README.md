### Install local environment
1. Install [pyenv](https://github.com/pyenv/pyenv)
2. Create and activate new env `pyenv virtualenv 3.10 tractorun && pyenv activate tractorun`
3. Install all dependencies: `pip install --extra-index-url https://artifactory.nebius.dev/artifactory/api/pypi/nyt/simple ."[all]`


### Build and push new runtime image
```shell
./run_build torchesaurus_runtime --push
```
Don't forget to update images in `run.py`

### Build and push new image for tests
```shell
./run_build torchesaurus_tests --push
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

### Bump library version
Just run `Create release` action.

### Build and upload a wheel
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
