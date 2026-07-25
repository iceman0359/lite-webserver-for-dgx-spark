#!/usr/bin/env sh
set -eu

cd "$(dirname "$0")"

RELEASE_BASE="${COMFYUI_MANAGER_RELEASE_BASE:-https://github.com/iceman0359/lite-webserver-for-dgx-spark/releases/latest/download}"
RELEASE_TAR="${RELEASE_BASE}/comfyui-manager-linux-arm64.tar.gz"
RELEASE_BINARY="comfyui-manager-linux-arm64"

download_file() {
  url="$1"
  dest="$2"

  if command -v curl >/dev/null 2>&1; then
    curl -fL --retry 3 --connect-timeout 10 -o "$dest" "$url"
    return
  fi

  if command -v wget >/dev/null 2>&1; then
    wget -O "$dest" "$url"
    return
  fi

  echo "curl or wget is required to download the prebuilt release." >&2
  return 1
}

install_prebuilt() {
  arch="$(uname -m)"
  case "$arch" in
    aarch64|arm64) ;;
    *)
      echo "This release is for Linux arm64/aarch64, but this machine reports: $arch" >&2
      return 1
      ;;
  esac

  if ! command -v tar >/dev/null 2>&1; then
    echo "tar is required to unpack the prebuilt release." >&2
    return 1
  fi

  tmp_dir="$(mktemp -d)"
  trap 'rm -rf "$tmp_dir"' EXIT INT TERM

  echo "No local comfyui-manager binary found. Downloading the Linux arm64 release..."
  download_file "$RELEASE_TAR" "$tmp_dir/release.tar.gz"
  tar -xzf "$tmp_dir/release.tar.gz" -C "$tmp_dir"

  if [ -f "$tmp_dir/${RELEASE_BINARY}.sha256" ] && command -v sha256sum >/dev/null 2>&1; then
    (cd "$tmp_dir" && sha256sum -c "${RELEASE_BINARY}.sha256")
  fi

  if command -v install >/dev/null 2>&1; then
    install -m 755 "$tmp_dir/$RELEASE_BINARY" ./comfyui-manager
  else
    cp "$tmp_dir/$RELEASE_BINARY" ./comfyui-manager
    chmod +x ./comfyui-manager
  fi

  rm -rf "$tmp_dir"
  trap - EXIT INT TERM
}

if [ ! -f config.json ]; then
  cp config.example.json config.json
fi

if [ ! -x ./comfyui-manager ]; then
  if ! install_prebuilt; then
    if command -v go >/dev/null 2>&1; then
      echo "Prebuilt release download failed. Building locally with Go..."
      sh ./build.sh
    else
      echo "Could not install the prebuilt release, and Go is not installed for a local build." >&2
      echo "Manual fallback: download comfyui-manager-linux-arm64.tar.gz from the GitHub Release and unpack it here." >&2
      exit 1
    fi
  fi
fi

exec ./comfyui-manager
