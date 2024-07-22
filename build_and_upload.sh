#!/usr/bin/env bash

set -ex


current_revision=$(git rev-parse HEAD)
current_tag=$(git describe --tags --exact-match $current_revision 2>/dev/null)

version_regex="^[0-9]+\.[0-9]+\.[0-9]+$"

if [[ ! $current_tag =~ $version_regex ]]; then
    echo "Called on a revision without version tag"
    exit 1
fi

pip wheel . -w dist --no-deps
twine upload -r local dist/tractorun-$current_tag-py3-none-any.whl
