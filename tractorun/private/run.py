from tractorun.private.run_internal import (
    run_local,
    run_tracto,
)
from tractorun.run import TractorunParams
from tractorun.run_info import RunInfo


def run_script(
    params: TractorunParams,
    local: bool,
) -> RunInfo:
    if local:
        return run_local(params)
    else:
        return run_tracto(params)
