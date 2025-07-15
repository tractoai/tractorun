#!/usr/bin/env bash

set -x

_CURRENT_DOCKER_TAG="2025-07-15-16-43-46"
IMAGE_GENERIC="ghcr.io/tractoai/tractorun-generic-tests:$_CURRENT_DOCKER_TAG"
IMAGE_TRACTORCH="ghcr.io/tractoai/tractorun-tractorch-tests:$_CURRENT_DOCKER_TAG"
IMAGE_TRACTORAX="ghcr.io/tractoai/tractorun-tractorax-tests:$_CURRENT_DOCKER_TAG"
IMAGE_TENSORPROXY="ghcr.io/tractoai/tractorun-tensorproxy-tests:$_CURRENT_DOCKER_TAG"
IMAGE_EXAMPLES="ghcr.io/tractoai/tractorun-examples-runtime:$_CURRENT_DOCKER_TAG"

PATH_GENERIC="/src/tests/generic"
PATH_TRACTORCH="/src/tests/tractorch"
PATH_TRACTORAX="/src/tests/tractorax"
PATH_TENSORPROXY="/src/tests/tensorproxy"
PATH_EXAMPLES="/src/tests/examples"

TEST_TYPE="${1:-all}"

IMAGES=()
TEST_PATHS=()

case "$TEST_TYPE" in
  "generic")
    IMAGES=("$IMAGE_GENERIC")
    TEST_PATHS=("$PATH_GENERIC")
    ;;
  "tractorch")
    IMAGES=("$IMAGE_TRACTORCH")
    TEST_PATHS=("$PATH_TRACTORCH")
    ;;
  "tractorax")
    IMAGES=("$IMAGE_TRACTORAX")
    TEST_PATHS=("$PATH_TRACTORAX")
    ;;
  "tensorproxy")
    IMAGES=("$IMAGE_TENSORPROXY")
    TEST_PATHS=("$PATH_TENSORPROXY")
    ;;
  "examples")
    IMAGES=("$IMAGE_EXAMPLES")
    TEST_PATHS=("$PATH_EXAMPLES")
    ;;
  "all")
    IMAGES=("$IMAGE_GENERIC" "$IMAGE_TRACTORCH" "$IMAGE_TRACTORAX" "$IMAGE_TENSORPROXY" "$IMAGE_EXAMPLES")
    TEST_PATHS=("$PATH_GENERIC" "$PATH_TRACTORCH" "$PATH_TRACTORAX" "$PATH_TENSORPROXY" "$PATH_EXAMPLES")
    ;;
  *)
    echo "Invalid test type: $TEST_TYPE. Choose from 'all', 'tensorproxy', or 'generic'."
    exit 1
    ;;
esac

run_tests() {
  local image=$1
  local test_path=$2
  docker run -it \
    --mount type=bind,source=.,target=/src \
    --network=host \
    -e YT_MODE=external \
    -e YT_PROXY="${YT_PROXY}" \
    -e YT_TOKEN="${YT_TOKEN}" \
    -e YT_CONFIG_PATCHES="${YT_CONFIG_PATCHES}" \
    -e YT_LOG_LEVEL="${YT_LOG_LEVEL}" \
    -e PYTHONPATH="/src:$PYTHONPATH" \
    -e PYTHONDONTWRITEBYTECODE=1 \
    "$image" \
    pytest "$test_path" "${@:3}"
}

for i in "${!IMAGES[@]}"; do
  if [[ "$2" == -* ]]; then
    TEST_NAME=""
    TEST_ARGS=${@:2}
  else
    TEST_NAME=$2
    TEST_ARGS=${@:3}
  fi
  run_tests "${IMAGES[$i]}" "${TEST_PATHS[$i]}/$TEST_NAME" $TEST_ARGS
done
