SCRIPT_DIR=$(dirname "$(realpath "$0")")
TRACTORUN_PATH=$(realpath "$SCRIPT_DIR/../../../tractorun")
_CURRENT_DOCKER_TAG="2025-02-25-18-24-08"
DOCKER_IMAGE=${DOCKER_IMAGE:-"ghcr.io/tractoai/tractorun-examples-runtime:$_CURRENT_DOCKER_TAG"}

tractorun \
    --yt-path "//tmp/$USER/$RANDOM}" \
    --bind-local-lib $TRACTORUN_PATH \
    --docker-image $DOCKER_IMAGE \
    --run-config-path config.yaml
