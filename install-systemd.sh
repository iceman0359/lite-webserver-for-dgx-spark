#!/usr/bin/env sh
set -eu

cd "$(dirname "$0")"

mkdir -p "$HOME/.config/systemd/user"
service_path="$HOME/.config/systemd/user/comfyui-manager.service"
install_dir="$HOME/Comfyui/comfyui-manager-lite"

mkdir -p "$install_dir"
current_dir="$(pwd)"
if [ "$current_dir" != "$install_dir" ]; then
  cp -R . "$install_dir/"
fi
chmod +x "$install_dir/start.sh" "$install_dir/build.sh"

sed "s#%h/Comfyui/comfyui-manager-lite#$install_dir#g" comfyui-manager.service > "$service_path"

systemctl --user daemon-reload
systemctl --user enable --now comfyui-manager.service

echo "Started: http://127.0.0.1:8848/"
