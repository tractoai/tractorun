import enum
import io
import os
import subprocess
from typing import (
    TYPE_CHECKING,
    NewType,
)

import attrs

from tractorun.sidecar import (
    RestartPolicy,
    Sidecar,
)


SidecarIndex = NewType("SidecarIndex", int)


class RestartVerdict(enum.IntEnum):
    skip = enum.auto()
    restart = enum.auto()
    fail = enum.auto()
    unknown = enum.auto()


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
        process = subprocess.Popen(
            sidecar.command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=1,
            universal_newlines=True,
            env=env,
            errors="replace",
        )
        if TYPE_CHECKING:
            assert isinstance(process.stdout, io.TextIOWrapper)
            assert isinstance(process.stderr, io.TextIOWrapper)
        os.set_blocking(process.stdout.fileno(), False)
        os.set_blocking(process.stderr.fileno(), False)
        return process

    @classmethod
    def run(cls, sidecar: Sidecar, env: dict[str, str]) -> "SidecarRun":
        process = cls._run_process(sidecar=sidecar, env=env)
        return SidecarRun(sidecar=sidecar, process=process, env=env)

    def poll(self) -> int | None:
        return self._process.poll()

    def wait(self) -> None:
        self._process.wait()

    def restart(self) -> "SidecarRun":
        self.terminate()
        return self.run(sidecar=self._sidecar, env=self._env)

    def terminate(self) -> None:
        self._process.terminate()

    def stdout(self) -> io.TextIOWrapper:
        if TYPE_CHECKING:
            assert isinstance(self._process.stdout, io.TextIOWrapper)
        return self._process.stdout

    def stderr(self) -> io.TextIOWrapper:
        if TYPE_CHECKING:
            assert isinstance(self._process.stderr, io.TextIOWrapper)
        return self._process.stderr

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
