# Backends

Backend is responsible for configuring `tractorun` to work with a specific ML framework, such as PyTorch or Jax.

## Tractorch

Sets up the distributed environment for PyTorch by `torch.distributed.init_process_group` with Tracto.ai specific.

### Dataset

Tractorch backend provides a custom [YtDataset](https://github.com/tractoai/tractorun/blob/main/tractorun/backend/tractorch/dataset.py#L32) class that extends `torch.utils.data.IterableDataset` and allows to effectively use YTSaurus table as a Dataset for PyTorch training.

The specifics of dataset usage depend on the training configuration, data storage format requirements, and the trainer being used. For example:
* If the primary process is responsible for reading the dataset and distributing the data to other processes (as in the case of libraries like [accelerate](https://huggingface.co/docs/accelerate/en/index) or trainers from [transformers](https://huggingface.co/docs/transformers/en/index)), the YtDataset can be used as is.
* In more complex cases, it is necessary to implement additional logic for reading data from the table, as demonstrated in our [nanotron fork](https://github.com/tractoai/tractorun/blob/main/tractorun/backend/tractorch/dataset.py#L32).

Additional example of using the `YtDataset` can be found on [GitHub](https://github.com/tractoai/tracto-examples/blob/main/notebooks/tractorun-llama-inference-tiny-stories-finetune.ipynb).

## Tractorax

Sets up the distributed environment for Jax by `jax.distributed.initialize` with Tracto.ai specific.

## Generic

Non-specialized backend, can be used as a basis for other backends or in case of non-coordinated operation.
