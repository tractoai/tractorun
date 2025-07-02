#!/usr/bin/env bash

set -x

SCRIPT_DIR=$(dirname "$(realpath "$0")")
TRACTORUN_PATH=$(realpath "$SCRIPT_DIR/..")
_CURRENT_DOCKER_TAG="2025-07-02-17-34-55"
DOCKER_IMAGE=${DOCKER_IMAGE:-"ghcr.io/tractoai/tractorun-examples-runtime:$_CURRENT_DOCKER_TAG"}


exec docker run -it \
  --mount type=bind,source=$TRACTORUN_PATH,target=$TRACTORUN_PATH \
  --network=host \
  -e DOCKER_IMAGE=$DOCKER_IMAGE \
  -e YT_MODE=external \
  -e YT_PROXY="${YT_PROXY}" \
  -e YT_TOKEN="${YT_TOKEN}" \
  -e WANDB_API_KEY="${WANDB_API_KEY}" \
  -e PYTHONPATH="$TRACTORUN_PATH:$PYTHONPATH" \
  $DOCKER_IMAGE \
  python3 "$(realpath "$1")" ${@:2}
