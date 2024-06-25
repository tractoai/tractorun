python3 ../../tractorun/cli/tractorun_runner.py \
    --nnodes 4 \
    --nproc_per_node 8 \
    --ngpu_per_proc 0 \
    --yt-path //home/gritukan/mnist/trainings/dense \
    --bind lightning_mnist_ddp_script.py:. \
    python3 lightning_mnist_ddp_script.py
