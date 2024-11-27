from collections import Counter
import json
import os
import shutil
from typing import Iterable
import warnings
import zipfile

import attrs
import yt.wrapper as yt
from yt.wrapper.errors import YtResolveError

from tractorun.bind import (
    BindCypress,
    BindLocal,
)
from tractorun.exception import TractorunConfigurationError


@attrs.define(kw_only=True, slots=True, auto_attribs=True)
class PackedBind:
    yt_path: str
    local_path: str


@attrs.define(kw_only=True, slots=True, auto_attribs=True)
class BindsPacker:
    _files: Iterable[BindLocal]
    _dirs: Iterable[BindLocal]

    @classmethod
    def from_binds(cls, binds: Iterable[BindLocal]) -> "BindsPacker":
        files: list[BindLocal] = []
        dirs: list[BindLocal] = []

        for bind in sorted(binds, key=lambda x: x.source):
            if os.path.isdir(bind.source):
                dirs.append(bind)
            else:
                files.append(bind)
        return BindsPacker(files=files, dirs=dirs)

    def _pack_file(self, archive_path: str, archive_name: str, source_path: str, destination_name: str) -> PackedBind:
        with zipfile.ZipFile(archive_path + ".zip", "w") as zipf:
            zipf.write(source_path, arcname=destination_name)
        return PackedBind(
            yt_path=f"{archive_name}.zip",
            local_path=f"{archive_path}.zip",
        )

    def _pack_dir(self, archive_path: str, archive_name: str, source_path: str) -> PackedBind:
        shutil.make_archive(archive_path, "zip", source_path)
        return PackedBind(
            yt_path=f"{archive_name}.zip",
            local_path=f"{archive_path}.zip",
        )

    def pack(self, base_result_path: str) -> list[PackedBind]:
        result: list[PackedBind] = []
        for idx, bind in enumerate(self._dirs):
            archive_name = f"__dir_{idx}"
            path = os.path.join(base_result_path, archive_name)
            packed_bind = self._pack_dir(
                archive_path=path,
                archive_name=archive_name,
                source_path=bind.source,
            )
            result.append(packed_bind)
        for idx, bind in enumerate(self._files):
            archive_name = f"__file_{idx}"
            path = os.path.join(base_result_path, archive_name)
            destination_name = os.path.split(bind.destination)[1]
            packed_bind = self._pack_file(
                archive_path=path,
                archive_name=archive_name,
                source_path=bind.source,
                destination_name=destination_name,
            )
            result.append(packed_bind)
        return result

    def unpack(self) -> None:
        # Runs inside a job
        for idx, bind in enumerate(self._dirs):
            path_w = f"__dir_{idx}.zip"
            with zipfile.ZipFile(path_w, "r") as zipf:
                zipf.extractall(path=bind.destination)
        for idx, bind in enumerate(self._files):
            path_w = f"__file_{idx}.zip"
            destination = os.path.split(bind.destination)[0]
            with zipfile.ZipFile(path_w, "r") as zipf:
                zipf.extractall(path=destination)

    def to_env(self) -> str:
        return json.dumps(
            {
                "files": [attrs.asdict(bind) for bind in self._files],  # type: ignore
                "dirs": [attrs.asdict(bind) for bind in self._dirs],  # type: ignore
            },
        )

    @classmethod
    def from_env(cls, content: str) -> "BindsPacker":
        parsed = json.loads(content)
        files = [BindLocal(source=record["source"], destination=record["destination"]) for record in parsed["files"]]
        dirs = [BindLocal(source=record["source"], destination=record["destination"]) for record in parsed["dirs"]]
        return BindsPacker(files=files, dirs=dirs)


@attrs.define(kw_only=True, slots=True, auto_attribs=True)
class PackedLib:
    archive_name: str
    path: str


@attrs.define(kw_only=True, slots=True, auto_attribs=True)
class BindsLibPacker:
    _paths: Iterable[str]

    def pack(self, base_result_path: str) -> list[PackedLib]:
        paths = [os.path.abspath(path) for path in self._paths]
        lib_to_original_path = {os.path.split(path)[1]: path for path in paths}
        non_uniq_names = [name for name, count in Counter(lib_to_original_path.keys()).items() if count > 1]
        assert len(non_uniq_names) == 0, f"Some libraries have the same names: {non_uniq_names}"

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


@attrs.define(kw_only=True, slots=True, auto_attribs=True)
class BindsCypressProcessor:
    _yt_client: yt.YtClient
    _binds: Iterable[BindCypress]

    def process(self) -> list[BindCypress]:
        new_binds: list[BindCypress] = []
        for bind in self._binds:
            node_type = self._get_source_node_type(bind.source)
            if node_type == "map_node":
                for file_name, yt_path in self._expand_nested_nodes(bind):
                    new_binds.append(
                        BindCypress(
                            source=yt_path,
                            destination=f"{bind.destination}/{file_name}",
                            attributes=bind.attributes,
                        )
                    )
            else:
                new_binds.append(bind)
        return new_binds

    def _get_source_node_type(self, path: str) -> str:
        try:
            return str(self._yt_client.get_attribute(path, "type"))
        except YtResolveError as e:
            raise TractorunConfigurationError(f"Source path for path {path} doesn't exist") from e

    def _expand_nested_nodes(self, bind: BindCypress) -> Iterable[tuple[str, str]]:
        nested_nodes = self._yt_client.list(bind.source)
        for nested_node in nested_nodes:
            nested_node_path = f"{bind.source}/{nested_node}"
            nested_node_type = self._get_source_node_type(nested_node_path)
            if nested_node_type != "file":
                warnings.warn(f"Skip {nested_node_path} because it is not a file, but {nested_node_type}")
                continue
            yield str(nested_node), str(nested_node_path)
