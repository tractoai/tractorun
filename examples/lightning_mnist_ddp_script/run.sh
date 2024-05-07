python3 ../../torchesaurus/cli/tractorun.py \
    --nnodes 4 \
    --nproc_per_node 8 \
    --ngpu_per_proc 0 \
    --path //home/gritukan/mnist/train \
    lightning_mnist_ddp_script.py
