# Options

## command

### cli
The command that will be executed by each worker on Tracto.

`tractorun python3 main.py`

### yaml
The command that will be executed by each worker on Tracto.
```yaml
command: ["python3", "main.py"]
```

### python
Callable python object, that will be pickled and executed by each worker on Tracto.
```python
from tractorun.run import run
from tractorun.toolbox import Toolbox
from tractorun.backend.tractorch import Tractorch

def train(toolbox: Toolbox):
  pass

run(train, backend=Tractorch())
```

## yt-path
Base directory on the cluster where `tractorun` stores its metadata. For `tractorch` backend, it also stores checkpoints used during operation.

By default, tractorun generates uniq folder in `//tmp/tractorun` for each run.

It is highly recommended that the same path be used for all iterations of training or inference for a particular model.

### cli
`tractorun --yt-path '//tmp/path'`

### yaml
```
yt_path: //tmp/path
```

### python
```python
from tractorun.run import run

run(yt_path="//tmp/path")
```

## run-config-path

Path to yaml file with `tractorun` config in cli-mode.

### cli
`tractorun --run-config-path ./config.yaml`

## docker-image
Docker image name to use for the job. Docker image should contain the same `tractorun` and `ytsaurus-client` versions.

### cli
`tractorun --docker-image my-docker-image`

### yaml
```yaml
docker_image: my-docker-image
```

### python
```python
from tractorun.run import run

run(docker_image="my-docker-image")
```

## title
Set the title for the operation. By default `tractorun` sets `Tractorun {yt-path}`.

### cli: `tractorun --title 'Operation Title'`

### yaml
```yaml
title: "Operation Title"
```

### python
```python
from tractorun.run import run

run(title="Operation Title")
```

## mesh.node-count
Number of nodes for training/inference. Default is 1.

### cli
`tractorun --mesh.node-count 3`

### yaml
```yaml
mesh:
  node_count: 3
```

### python
```python
from tractorun.run import run
from tractorun.mesh import Mesh

run(mesh=Mesh(node_count=3))
```

## mesh.process-per-node
Number of processes per node. Default is 1.

### cli
`tractorun --mesh.process-per-node 2`

### yaml
```yaml
mesh:
  process_per_node: 2
```

### python
```python
from tractorun.run import run
from tractorun.mesh import Mesh

run(mesh=Mesh(process_per_node=2))
```

## mesh.gpu-per-process
Number of GPUs per process. Default is 0.

### cli
`tractorun --mesh.gpu-per-process 1`

### yaml
```yaml
mesh:
  gpu_per_process: 1
```

### python
```python
from tractorun.run import run
from tractorun.mesh import Mesh

run(mesh=Mesh(gpu_per_process=1))
```

### mesh.pool-trees
Typically, the default pool tree contains only CPU resources. You need to specify the pool tree name to use GPU resources.

cli: `tractorun --mesh.pool-trees "gpu_h100"`

yaml:
```yaml
mesh:
  pool_trees: [gpu_h100]
```

### python
```python
from tractorun.run import run
from tractorun.mesh import Mesh

run(mesh=Mesh(pool_trees=["gpu_h100"]))
```

## mesh.pool
For more flexible resource management, it is possible to specify specific [pools](https://ytsaurus.tech/docs/en/user-guide/data-processing/scheduler/scheduler-and-pools#pools-and-pool-trees).

### cli
`tractorun --mesh.pool research`

### yaml
```yaml
mesh:
  pool: research
```

### python
```python
from tractorun.run import run
from tractorun.mesh import Mesh

run(mesh=Mesh(pool="research"))
```

## env
Set an environment variable in process.

Environment variables can be set by value or from a Cypress node. Environment variables from Cypress nodes can be used to provide secrets given that ACLs are properly configured.

### cli
`tractorun --env '{"name": "MY_VAR_1", "value": "123"}' --env '{"name": "ENV_VAR_2, "cypress_path": "//tmp/sec_value"}'`

### yaml
```yaml
env:
  - name: MY_VAR_1
    value: "123"
  - name: MY_VAR_2
    value: "//tmp/sec_value"
```

### python
```python
from tractorun.run import run
from tractorun.env import EnvVariable

run(
    env=[
        EnvVariable(
            name="MY_VAR_1",
            value="123",
        ),
        EnvVariable(
            name="MY_VAR_2",
            cypress_path="//tmp/sec_value",
        ),
    ],
)
```

## resources.cpu-limit
CPU limit for the node.

### cli
`tractorun --resources.cpu-limit 4`

### yaml
```yaml
resources:
  cpu_limit: 4
```

### python
```python
from tractorun.run import run
from tractorun.resources import Resources

run(
    resources=Resources(
        cpu_limit=4,
    ),
)
```

## resources.memory-limit
Memory limit for the node in bytes.

### cli
`tractorun --resources.memory-limit 17179869184`

### yaml
```yaml
resources:
  memory_limit: 17179869184  # 16GiB
```

### python
```python
from tractorun.run import run
from tractorun.resources import Resources

run(
    resources=Resources(
        memory_limit=17179869184,
    ),
)
```

## user-config
A JSON config to be passed to a container. To read the config inside a process use `get_user_config` method of `toolbox`:

```python
user_config = toolbox.get_user_config()
print(user_config["a"])
```

### cli
`tractorun --user-config '{"a": "b"}'`

### yaml
```yaml
user_config:
  a: "b"
```

### python
```python
from tractorun.run import run

run(
    user_config={"a": "b"},
)
```

## bind-local
Bind a local file or folder to be passed to the container on Tracto.ai platform. Format:
* simple `./local_path:/path/in/container`
* json: `{"source": "./local_path", "destination": "/path/in/container"}`

### cli
`tractorun --bind-local /path/in/container --bind-local '{"source": "./local_path", "destination": "/path/in/container"}'`

## yaml
```yaml
bind_local: 
  - source: ./local_path
    destination: /path/in/container
```

### python
```python
from tractorun.run import run
from tractorun.bind import BindLocal

run(
    binds_local=[
        BindLocal(
            source="./local_path",
            destination="/path/in/container",
        ),
    ],
)
```

## bind-local-lib
The path to a local Python library to bind it to the remote container and set the remote `PYTHONPATH`. Useful for development purposes. In the production environment, prepare the environment using docker images.

Warning: 
* works only for pure-python libraries
* dependencies won't be installed
* `pip install` won't be called

### cli
`tractorun --bind-local-lib /path/to/local/lib`

### yaml
```yaml
bind_local_lib:
  - "/path/to/local/lib"
```

### python
```python
from tractorun.run import run

run(
    binds_local_lib=["/path/to/local/lib"],
)
```

## bind-cypress
Bind a cypress path to be passed to the container on Tracto.ai platform. Format:
* simple `//tmp/path:/path/in/container`
* json: `{"source": "//tmp/path", "destination": "/path/in/container", "attributes": {"executable": true, "format": null, "bypass_artifact_cache": false}}`

[Attributes](https://ytsaurus.tech/docs/en/user-guide/data-processing/operations/operations-options#files) are optional.

Bind cypress supports:
* Files.
* Map nodes.
* Symlinks to files and map nodes.

Please be careful when using symlinks due to circular links.

### cli
`tractorun --bind-cypress '//tmp/path:/path/in/container' --bind-cypress '{"source": "//tmp/path", "destination": "/path/in/container", "attributes": {"executable": true, "format": null, "bypass_artifact_cache": false}}'`

### yaml
```yaml
bind_cypress: 
  - source: //tmp/path
    destination: /path/in/container
    attributes:
      executable: True
      format: None
      bypass_artifact_cache: False
```

### python
```python
from tractorun.run import run
from tractorun.bind import BindCypress, BindAttributes

run(
    binds_local=[
        BindCypress(
            source="//tmp/path",
            destination="/path/in/container",
            attributes=BindAttributes(
                executable=True,
                format=None,
                bypass_artifact_cache=False,
            ),
        ),
    ],
)
```

## sidecar
Specify a sidecar in JSON format.


The `sidecar` option in `tractorun` is conceptually similar to the [sidecar container](https://kubernetes.io/docs/concepts/workloads/pods/sidecar-containers/) in k8s. It allows to set up extra processes that run next to the main processes on each node in the same environment. Use cases:

* A sidecar can send logs to systems like Victoria Metrics for monitoring.
* A sidecar can run tensorproxy for jax-based trainings.
* A sidecar can be used for asynchronous checkpoint saving.

Restart policy:
* `always`
    * always restart a sidecar
* `on_failure`
  * restart a sidecar on failure, don't restart on success
* `never`
  * newer restart a sidecar
* `fail`
  * fail the entire operation in case of a sidecar's fail

### cli
`tractorun --sidecar '{"command": ["python", "script.py"], "restart_policy": "always"}'`

### yaml
```yaml
sidecars:
  - command: ["python", "script.py"]
    restart_policy: always
```

### python
```python
from tractorun.run import run
from tractorun.sidecar import Sidecar, RestartPolicy

run(
    sidecars=[
        Sidecar(
            command=["python", "script.py"],
            restart_policy=RestartPolicy.ALWAYS,
        ),
    ],
)
```

## proxy-stderr-mode
Proxy job stderr to the terminal. Modes:
* `disabled`
* `primary` - proxy all primary-process logs to the local terminal.

### cli
`tractorun --proxy-stderr-mode primary`

### yaml
```yaml
proxy_stderr_mode: primary
```

### python
```python
from tractorun.run import run
from tractorun.stderr_reader import StderrMode

run(
    proxy_stderr_mode=StderrMode.primary,
)
```

## operation-log-mode
Beta.

Set the operation's log mode. Modes:
* `default`
* `realtime_yt_table` - writes logs of each worker and sidecar to `<yt-path>/logs`

### cli
`tractorun --operation-log-mode realtime_yt_table`

### yaml
```yaml
operation_log_mode: realtime_yt_table
```

### python
```python
from tractorun.run import run
from tractorun.operation_log import OperationLogMode

run(
    operation_log_mode=OperationLogMode.realtime_yt_table,
)
```

## no-wait
Do not create a transaction and do not wait for the operation to complete. Can be useful for running long operations to avoid problems with network connectivity between the local host and Tracto.ai platform.

### cli
`tractorun --no-wait`

### yaml
```yaml
no_wait: true
```

### python
```python
from tractorun.run import run

run(
    no_wait=True,
)
```

## dry-run
Get internal information without actually running the operation.

### cli
`tractorun --dry-run`

### yaml
```yaml
dry_run: true
```

### python
```python
from tractorun.run import run

run(
    dry_run=True,
)
```

## docker-auth-secret.cypress-path
Required for authenticating with private Docker registries. To write a document and set the appropriate ACL use:

```python
import yt.wrapper as yt

user = "current_user"

acl = [  
    {
        "action": "allow",
        "subjects": [
            user,
        ],
        "permissions": [
            "read",
            "write",
            "administer",
            "remove",
            "manage",
            "modify_children"
        ],
        "inheritance_mode": "object_and_descendants"
    },
]

yt.create(
    "document",
    "//tmp/sec_path",
    attributes={"acl": acl},
)
yt.set(
    "//tmp/sec_path",
    {
        "secrets": {
            "username": "username",
            "password": "password",
        },
    },
)
```

Path to Cypress secret with [docker auth](https://ytsaurus.tech/docs/en/user-guide/data-processing/layers/layer-paths#docker_auth) `{"username": "placeholder", "password": "placeholder", "auth": "placeholder"}`.

### cli
`tractorun --docker-auth-secret.cypress-path //path/to/secret`

### yaml
```yaml
docker_auth_secret:
  cypress_path: //path/to/secret
```

### python
```python
from tractorun.run import run
from tractorun.docker_auth import DockerAuthSecret

run(
    docker_auth=DockerAuthSecret(
        cypress_path="//path/to/secret",
    ),
)
```

## cluster-config-path
Path to the global tractorun configuration. Default is `//home/tractorun/config`.

cli: `tractorun --cluster-config-path //home/tractorun/config`

yaml:
```yaml
cluster_config_path: //home/tractorun/config
```

### python
```python
from tractorun.run import run

run(
    cluster_config_path="//home/tractorun/config",
)
```

## yt-operation-spec
YTSaurus [operation specification](https://ytsaurus.tech/docs/en/user-guide/data-processing/operations/operations-options). Please do not use this option without approval from your system administrator.

### cli
`tractorun --yt-operation-spec '{"title": "custom title"}'`

### yaml
```yaml
yt_operation_spec:
  title: "custom title"
```

### python
```python
from tractorun.run import run

run(
    yt_operation_spec={"title": "custom title"},
)
```

## yt-task-spec
YTSaurus [task specification](https://ytsaurus.tech/docs/en/user-guide/data-processing/operations/vanilla). Please do not use this option without approval from your system administrator.

### cli
`tractorun --yt-task-spec '{"title": "custom title"}'`

### yaml
```yaml
yt_task_spec:
  title: "custom title"
```

### python
```python
from tractorun.run import run

run(
    yt_task_spec={"title": "custom title"},
)
```

## local
Beta.

Run code on a local host, without creation operation on Tracto.ai platform. Useful for debugging.

### cli
`tractorun --local True`

### yaml
```yaml
local: true
```

### python
```python
from tractorun.run import run

run(
    local=True,
)
```
