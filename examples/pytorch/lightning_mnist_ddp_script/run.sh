tractorun \
    --mesh.node-count 1 \
    --mesh.process-per-node 4 \
    --mesh.gpu-per-process 0 \
    --resources.memory-limit 8076021002 \
    --yt-path //tmp/$USER/$RANDOM \
    --user-config '{"dataset_path": "//home/samples/mnist-torch-train"}' \
    --bind-local './script.py:/script.py' \
    --bind-local-lib '../../../tractorun' \
    --docker-image $DOCKER_IMAGE \
    python3 /script.py
