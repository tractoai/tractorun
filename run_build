#!/usr/bin/env bash

set -x

ALL_BAKE_FILES=(
    vars.hcl
    targets.hcl
)

FILE_ARGS=()
for BAKE_FILE in "${ALL_BAKE_FILES[@]}"
do
  FILE_ARGS+=(--file "docker/${BAKE_FILE}")
done

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

export PROJECT_ROOT="${SCRIPT_DIR}"
export DOCKER_TAG=$(date '+%Y-%m-%d-%H-%M-%S')  # TODO: allow to define from args

exec docker buildx bake "${FILE_ARGS[@]}" "$@"
