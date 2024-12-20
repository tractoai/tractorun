#!/bin/bash

NEW_TAG=$1

if [[ -z "$NEW_TAG" ]]; then
  echo "Usage: $0 new_tag"
  exit 1
fi

LC_ALL=C.UTF-8 find . -type f -name "*.sh" -exec sed -i '' -E "s/^(_CURRENT_DOCKER_TAG=).*/\1\"$NEW_TAG\"/" {} +
LC_ALL=C.UTF-8 find . -type f -name "*.py" -exec sed -i '' -E "s/^(_CURRENT_DOCKER_TAG = ).*/\1\"$NEW_TAG\"/" {} +
LC_ALL=C.UTF-8 sed -i '' -E "s|--docker-image ghcr.io/tractoai/tractorun-examples-runtime:[^ ]*|--docker-image ghcr.io/tractoai/tractorun-examples-runtime:$NEW_TAG|" ./README.md

echo "Tag updated successfully in all files."
