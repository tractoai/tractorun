import os
import tarfile

import attrs


@attrs.define(kw_only=True, slots=True, auto_attribs=True)
class Bind:
    source: str
    destination: str


@attrs.define(kw_only=True, slots=True, auto_attribs=True)
class BindsPacker:
    _binds: list[Bind]

    def pack(self) -> list[str]:
        binds = sorted(self._binds, key=lambda b: b.source)
        paths = []
        for idx, bind in enumerate(binds):
            path = f".binds/{idx}.tar"
            with tarfile.open(path, "w:gz") as tar:
                tar.add(bind.source, arcname=os.path.basename(bind.source))
                paths.append(path)
        return paths

    def unpack(self) -> None:
        binds = sorted(self._binds, key=lambda b: b.source)
        for idx_w, bind_w in enumerate(binds):
            path_w = f".binds/{idx_w}.tar"
            with tarfile.open(path_w, "r:gz") as tar_w:
                tar_w.extractall(path=bind_w.destination)
