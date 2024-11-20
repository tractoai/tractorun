tractorun \
    --mesh.node-count 1 \
    --mesh.process-per-node 4 \
    --mesh.gpu-per-process 0 \
    --resources.memory-limit 8076021002 \
    --yt-path ${SAMPLE_YT_PATH:-//tmp/$USER/$RANDOM} \
    --user-config "{'dataset_path': ${SAMPLE_YT_PATH:"//home/samples/mnist-torch-train"}}" \
    --bind-local './lightning_mnist_ddp_script.py:/lightning_mnist_ddp_script.py' \
    --bind-local-lib '../../../tractorun' \
    --docker-image $DOCKER_IMAGE \
    python3 /lightning_mnist_ddp_script.py
