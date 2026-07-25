#!/usr/bin/env sh
set -eu

go build -trimpath -ldflags="-s -w" -o comfyui-manager .
