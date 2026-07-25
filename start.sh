#!/usr/bin/env sh
set -eu

cd "$(dirname "$0")"

if [ ! -f config.json ]; then
  cp config.example.json config.json
fi

if [ ! -x ./comfyui-manager ]; then
  if ! command -v go >/dev/null 2>&1; then
    echo "Go is required for the first build. Install Go or place ./comfyui-manager here." >&2
    exit 1
  fi
  sh ./build.sh
fi

exec ./comfyui-manager
