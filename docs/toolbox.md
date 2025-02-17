# Toolbox

`tractorun.toolbox.Toolbox` provides extra integrations with the Tracto.ai platform.

## Toolbox.yt_client

Ready-to-use [YTsaurus client](https://ytsaurus.tech/docs/en/api/python/start). Preconfigured to work with the current cluster using the credentials of the user who launched the tractorun operation.

## Toolbox.description_manager

Provides the ability to display additional information on the operation page in the UI, such as the current training step.

### Toolbox.description_manager.set

Sets an arbitrary key-value structure as the description of the operation.

```python
toolbox.description_manager.set({"training_step": training_step})
```

### Toolbox.description_manager.make_cypress_link

Generates a link to cypress path.

```python
dataset_link = toolbox.description_manager.make_cypress_link("//tmp/dataset")
toolbox.description_manager.set({"dataset": dataset_link})
```

## Toolbox.coordinator

Contains information about the state of a distributed operation, such as the address of the training's primary host, incarnation id, etc. It can be useful for implementing custom Datasets or Checkpoints.

### Toolbox.coordinator.get_self_index

Returns uniq index of the current process in the distributed operation, which is unique across all nodes.

### Toolbox.coordinator.get_total_peer_count

Returns the total number of processes in the distributed operation.

### Toolbox.coordinator.get_incarnation_id

Returns the unique identifier of the current incarnation of the distributed operation. The incarnation id increments by 1 with each new operation run in the same `yt_path`, whether the run is successful or unsuccessful.

### Toolbox.coordinator.is_primary

Returns `True` if the current process is the primary process in the distributed operation.

### Toolbox.coordinator.get_primary_endpoint

Returns the address of the primary process in the distributed operation.

### Toolbox.coordinator.get_process_index

Returns uniq index of the current process in the current node. It can be useful if a library or framework requires explicitly specifying the id of the GPU being used.

### Toolbox.coordinator.get_self_endpoint

Returns the hostname of the current job.

## Toolbox.checkpoint_manager

*Beta* version of the checkpoint manager. You can find an example on [GitHub](https://github.com/tractoai/tractorun/blob/main/examples/pytorch/torch_mnist_checkpoints/torch_mnist_checkpoints.py).

### Toolbox.checkpoint_manager.get_last_checkpoint

Find and read the last checkpoint for the current `yt_path`.

### Toolbox.checkpoint_manager.save_checkpoint

Save a checkpoint as binary file on YTSaurus.

## Toolbox.mesh

Provides the information about current [Mesh configuration](https://github.com/tractoai/tractorun/blob/main/docs/options.md#meshnode-count).
