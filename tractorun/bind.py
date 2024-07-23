from collections import Counter
import json
import os
import shutil
import zipfile

import attrs


@attrs.define(kw_only=True, slots=True, auto_attribs=True)
class Bind:
    source: str
    destination: str


@attrs.define(kw_only=True, slots=True, auto_attribs=True)
class PackedBind:
    archive_name: str
    path: str


@attrs.define(kw_only=True, slots=True, auto_attribs=True)
class BindsPacker:
    _binds: list[Bind]

    def pack(self, base_result_path: str) -> list[PackedBind]:
        binds = sorted(self._binds, key=lambda b: b.source)
        result: list[PackedBind] = []
        for idx, bind in enumerate(binds):
            archive_name = f"__{idx}"
            path = os.path.join(base_result_path, archive_name)
            root_dir, base_dir = os.path.split(os.path.abspath(bind.source))
            shutil.make_archive(path, "zip", root_dir, base_dir)
            result.append(
                PackedBind(
                    archive_name=f"{archive_name}.zip",
                    path=f"{path}.zip",
                ),
            )
        return result

    def unpack(self) -> None:
        binds = sorted(self._binds, key=lambda b: b.source)
        for idx, bind in enumerate(binds):
            path_w = f"__{idx}.zip"
            with zipfile.ZipFile(path_w, "r") as zipf:
                zipf.extractall(path=bind.destination)

    def to_env(self) -> str:
        # sorry
        return json.dumps([attrs.asdict(bind) for bind in self._binds])  # type: ignore

    @classmethod
    def from_env(cls, content: str) -> "BindsPacker":
        # sorry
        parsed = json.loads(content)
        binds = [Bind(source=record["source"], destination=record["destination"]) for record in parsed]
        return BindsPacker(binds=binds)


@attrs.define(kw_only=True, slots=True, auto_attribs=True)
class PackedLib:
    archive_name: str
    path: str


@attrs.define(kw_only=True, slots=True, auto_attribs=True)
class BindsLibPacker:
    _paths: list[str]

    def pack(self, base_result_path: str) -> list[PackedLib]:
        paths = [os.path.abspath(path) for path in self._paths]
        lib_to_original_path = {os.path.split(path)[1]: path for path in paths}
        non_uniq_names = [name for name, count in Counter(lib_to_original_path.keys()).items() if count > 1]
        assert len(non_uniq_names) == 0, f"Some libraries has the same names: {non_uniq_names}"

        result: list[PackedLib] = []
        for lib_name, original_path in lib_to_original_path.items():
            path = os.path.join(base_result_path, f"__{lib_name}")
            root_dir, base_dir = os.path.split(os.path.abspath(original_path))
            shutil.make_archive(path, "zip", root_dir, base_dir)
            result.append(
                PackedLib(
                    archive_name=lib_name,
                    path=f"{path}.zip",
                )
            )
        return result
