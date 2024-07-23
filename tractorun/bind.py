from collections import Counter
import json
import os
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
            archive_name = f"{idx}.zip"
            path = os.path.join(base_result_path, archive_name)
            with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zipf:
                for root, dirs, files in os.walk(bind.source):
                    for d in dirs:
                        dir_path = os.path.join(root, d)
                        arcname = os.path.relpath(dir_path, bind.source)
                        zipf.writestr(arcname + os.path.sep, "")
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, bind.source)
                        zipf.write(file_path, arcname=arcname)
                result.append(
                    PackedBind(
                        archive_name=archive_name,
                        path=path,
                    ),
                )
        return result

    def unpack(self) -> None:
        binds = sorted(self._binds, key=lambda b: b.source)
        for idx, bind in enumerate(binds):
            path_w = f"{idx}.zip"
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
            path = os.path.join(base_result_path, f"{lib_name}.zip")
            with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zipf:
                zipf.writestr(lib_name + os.path.sep, "")
                for root, dirs, files in os.walk(original_path):
                    for d in dirs:
                        dir_path = os.path.join(root, d)
                        arcname = os.path.join(
                            lib_name,
                            os.path.relpath(dir_path, original_path),
                        )
                        zipf.writestr(arcname + os.path.sep, "")
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.join(
                            lib_name,
                            os.path.relpath(file_path, original_path),
                        )
                        zipf.write(file_path, arcname=arcname)
                result.append(
                    PackedLib(
                        archive_name=lib_name,
                        path=path,
                    )
                )
        return result
