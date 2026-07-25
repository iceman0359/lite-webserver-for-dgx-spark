# lite-webserver-for-dgx-spark

Lightweight local-only backend for an aarch64 Ubuntu ComfyUI container manager.

## Run On Ubuntu aarch64 Without Go

Recommended install:

```bash
mkdir -p ~/Comfyui/webserver
cd ~/Comfyui/webserver
wget -O comfyui-manager-linux-arm64.tar.gz \
  https://github.com/iceman0359/lite-webserver-for-dgx-spark/releases/latest/download/comfyui-manager-linux-arm64.tar.gz
tar -xzf comfyui-manager-linux-arm64.tar.gz
cp comfyui-manager-linux-arm64 comfyui-manager
chmod +x comfyui-manager start.sh
./start.sh
```

If you cloned the source repository instead, `start.sh` will download the latest Linux arm64 release automatically when `./comfyui-manager` is missing. Go is only needed when you want to build from source yourself.

## Build From Source On Ubuntu aarch64

```bash
cd comfyui-manager-lite
cp config.example.json config.json
sh build.sh
./comfyui-manager
```

Or use the startup script:

```bash
sh start.sh
```

Install as a user-level systemd service:

```bash
sh install-systemd.sh
systemctl --user status comfyui-manager.service
```

Open:

```text
http://127.0.0.1:8848/
```

The server binds to `127.0.0.1` by default, so it is only reachable from the local machine.

## Backend Design

- Single Go binary, no Node/Python runtime.
- Uses only Go standard library.
- Static UI is embedded into the binary.
- Terminal/log output uses bounded ring buffers, so output does not grow forever in memory.
- User terminals run through a lightweight Linux PTY, so interactive bash, ANSI colors, completion, and control keys behave closer to an Ubuntu terminal.
- Download tasks run through one serial worker to avoid multiple large downloads competing for memory and disk IO.
- File listing skips hidden files and caps large directory responses.
- File manager and downloader paths are restricted to the two managed roots only:
  `/home/simonsyoyo/Comfyui/comfy_container_data` and `/home/simonsyoyo/Comfyui/ComfyUI/models`.
- Docker container actions only call `docker start`, `docker stop`, `docker restart`, and `docker logs`.
- File deletion maps host data paths into the Docker container and runs `docker exec comfyui rm -rf -- <container-path>`.
- Clash/Mihomo mode switching uses the `external-controller` API.
- Clash Verge start/stop uses configurable local commands. The system proxy switch uses Ubuntu `gsettings` when available and also tries to update Clash Verge Rev `verge.yaml`.

## Important Defaults

```text
Data root:
/home/simonsyoyo/Comfyui/comfy_container_data

Default subfolders:
custom_nodes
output
input
user

Container data root:
/opt/ComfyUI
```

Host delete path example:

```text
/home/simonsyoyo/Comfyui/comfy_container_data/output/a.png
```

Container delete path:

```text
/opt/ComfyUI/output/a.png
```

## API Overview

```text
GET  /api/health
GET  /api/config
POST /api/config

GET  /api/container/status
POST /api/container/start
POST /api/container/stop
POST /api/container/restart
POST /api/container/logs/start

GET  /api/files/list?path=/some/path
POST /api/files/delete

GET  /api/download/dirs

GET  /api/terminal/sessions
POST /api/terminal/create
POST /api/terminal/docker-bash
POST /api/terminal/input
POST /api/terminal/control
POST /api/terminal/close
GET  /api/terminal/stream?id=downloads

POST /api/download/start

GET  /api/clash/state
POST /api/clash/detect
POST /api/clash/mode
POST /api/clash/proxy
POST /api/clash/start
POST /api/clash/stop

POST /api/service/restart
POST /api/service/stop
```

## Notes

The terminal implementation uses a small Linux PTY path without Node/Python. It is intended for ordinary shell work, downloads, logs, and Docker bash basics.
