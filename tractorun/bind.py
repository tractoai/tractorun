import json
import os
import shutil
import tarfile

import attr


@attr.define(kw_only=True, slots=True, auto_attribs=True)
class Bind:
    source: str
    destination: str


@attr.define(kw_only=True, slots=True, auto_attribs=True)
class BindsPacker:
    _binds: list[Bind]

    def pack(self) -> list[str]:
        # TODO: use tmp
        if os.path.exists(".binds"):
            shutil.rmtree(".binds")
        os.mkdir(".binds")

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

    def to_env(self) -> str:
        # sorry
        return json.dumps([attr.asdict(bind) for bind in self._binds])  # type: ignore

    @classmethod
    def from_env(cls, content: str) -> "BindsPacker":
        # sorry
        parsed = json.loads(content)
        binds = [Bind(source=record["source"], destination=record["destination"]) for record in parsed]
        return BindsPacker(binds=binds)
