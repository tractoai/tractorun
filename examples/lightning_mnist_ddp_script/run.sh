python3 ../../tractorun/cli/tractorun.py \
    --nnodes 4 \
    --nproc_per_node 8 \
    --ngpu_per_proc 0 \
    --yt-path //home/gritukan/mnist/trainings/dense \
    lightning_mnist_ddp_script.py
