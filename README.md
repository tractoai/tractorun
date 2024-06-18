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

### Run tests
```shell
YT_PROXY=dirac.yt.nemax.nebiuscloud.net YT_TOKEN=<...> ./run_tests . -s
```
