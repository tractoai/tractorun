SCRIPT_DIR=$(dirname "$(realpath "$0")")
TRACTORUN_PATH=$(realpath "$SCRIPT_DIR/../../../tractorun")
DOCKER_IMAGE=${DOCKER_IMAGE:-"cr.ai.nebius.cloud/crnf2coti090683j5ssi/tractorun/examples_runtime:2024-11-20-20-00-05"}

tractorun \
    --mesh.node-count 1 \
    --mesh.process-per-node 4 \
    --mesh.gpu-per-process 0 \
    --resources.memory-limit 8076021002 \
    --yt-path "//tmp/$USER/$RANDOM}" \
    --user-config '{"dataset_path": "//home/samples/mnist-torch-train"}' \
    --bind-local './lightning_mnist_ddp_script.py:/lightning_mnist_ddp_script.py' \
    --bind-local-lib $TRACTORUN_PATH \
    --docker-image $DOCKER_IMAGE \
    python3 /lightning_mnist_ddp_script.py
