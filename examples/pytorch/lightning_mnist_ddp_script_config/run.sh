SCRIPT_DIR=$(dirname "$(realpath "$0")")
TRACTORUN_PATH=$(realpath "$SCRIPT_DIR/../../../tractorun")
DOCKER_IMAGE=${DOCKER_IMAGE:-"cr.ai.nebius.cloud/crnf2coti090683j5ssi/tractorun/examples_runtime:2024-11-20-20-00-05"}

tractorun \
    --yt-path "//tmp/$USER/$RANDOM}" \
    --bind-local-lib $TRACTORUN_PATH \
    --docker-image $DOCKER_IMAGE \
    --run-config-path config.yaml
