SCRIPT_DIR=$(dirname "$(realpath "$0")")
TRACTORUN_PATH=$(realpath "$SCRIPT_DIR/../../../tractorun")
_CURRENT_DOCKER_TAG="2025-09-30-15-26-34"
DOCKER_IMAGE=${DOCKER_IMAGE:-"ghcr.io/tractoai/tractorun-examples-runtime:$_CURRENT_DOCKER_TAG"}

tractorun \
    --mesh.node-count 1 \
    --mesh.process-per-node 4 \
    --mesh.gpu-per-process 0 \
    --resources.memory-limit 8076021002 \
    --yt-path "//tmp/$USER/$RANDOM}" \
    --user-config '{"dataset_path": "//home/samples/mnist-torch-train"}' \
    --bind-local './lightning_mnist_ddp_script_config.py:/lightning_mnist_ddp_script_config.py' \
    --bind-local-lib $TRACTORUN_PATH \
    --docker-image $DOCKER_IMAGE \
    python3 /lightning_mnist_ddp_script_config.py
