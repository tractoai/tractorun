import enum
import subprocess
import sys
from typing import Optional

import attrs


class RestartPolicy(str, enum.Enum):
    ON_FAILURE = "on_failure"
    ALWAYS = "always"
    NEVER = "never"
    FAIL = "fail"


class RestartVerdict(enum.IntEnum):
    skip = enum.auto()
    restart = enum.auto()
    fail = enum.auto()
    unknown = enum.auto()


@attrs.define(kw_only=True, slots=True, auto_attribs=True)
class Sidecar:
    command: list[str]
    restart_policy: RestartPolicy


@attrs.define(kw_only=True, slots=True, auto_attribs=True)
class SidecarRunner:
    _command: list[str]
    _env: dict[str, str]

    def run(self) -> subprocess.Popen:
        process = subprocess.Popen(
            self._command,
            stdout=sys.stderr,
            stderr=sys.stderr,
            bufsize=1,
            universal_newlines=True,
            env=self._env,
        )
        return process


@attrs.define(kw_only=True, slots=True, auto_attribs=True)
class SidecarRun:
    _restart_policy: RestartPolicy
    _runner: SidecarRunner
    _process: subprocess.Popen

    def poll(self) -> Optional[int]:
        return self._process.poll()

    def restart(self) -> None:
        assert self._process.poll() is None
        self._process = self._runner.run()

    def terminate(self) -> None:
        self._process.terminate()

    def need_restart(self) -> RestartVerdict:
        exit_code = self._process.poll()
        if exit_code is None:
            return RestartVerdict.skip
        match self._restart_policy:
            case RestartPolicy.ON_FAILURE:
                if exit_code == 0:
                    return RestartVerdict.skip
                return RestartVerdict.restart
            case RestartPolicy.ALWAYS:
                return RestartVerdict.restart
            case RestartPolicy.NEVER:
                return RestartVerdict.skip
            case RestartPolicy.FAIL:
                return RestartVerdict.fail
            case _:
                return RestartVerdict.unknown
