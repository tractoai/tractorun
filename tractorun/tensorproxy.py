import attrs

from tractorun.sidecar import RestartPolicy


@attrs.define(kw_only=True, slots=True, auto_attribs=True)
class TensorproxySidecar:
    enabled: bool
    restart_policy: RestartPolicy
    yt_path: str
