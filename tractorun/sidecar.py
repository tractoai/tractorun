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
class SidecarRun:
    _sidecar: Sidecar
    _process: subprocess.Popen
    _env: dict[str, str]

    @property
    def command(self) -> list[str]:
        return self._sidecar.command

    @classmethod
    def _run_process(cls, sidecar: Sidecar, env: dict[str, str]) -> subprocess.Popen:
        return subprocess.Popen(
            sidecar.command,
            stdout=sys.stderr,
            stderr=sys.stderr,
            bufsize=1,
            universal_newlines=True,
            env=env,
        )

    @classmethod
    def run(cls, sidecar: Sidecar, env: dict[str, str]) -> "SidecarRun":
        process = cls._run_process(sidecar=sidecar, env=env)
        return SidecarRun(sidecar=sidecar, process=process, env=env)

    def poll(self) -> Optional[int]:
        return self._process.poll()

    def wait(self) -> None:
        self._process.wait()

    def restart(self) -> None:
        self.terminate()
        self._process = self._run_process(sidecar=self._sidecar, env=self._env)

    def terminate(self) -> None:
        self._process.terminate()

    def need_restart(self) -> RestartVerdict:
        exit_code = self._process.poll()
        if exit_code is None:
            return RestartVerdict.skip
        match self._sidecar.restart_policy:
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
