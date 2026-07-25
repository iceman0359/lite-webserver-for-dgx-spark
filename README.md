# lite-webserver-for-dgx-spark

Lightweight local-only backend for an aarch64 Ubuntu ComfyUI container manager.

## Build On Ubuntu aarch64

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
- Download tasks run through one serial worker to avoid multiple large downloads competing for memory and disk IO.
- File listing skips hidden files and caps large directory responses.
- Docker container actions only call `docker start`, `docker stop`, `docker restart`, and `docker logs`.
- File deletion maps host data paths into the Docker container and runs `docker exec comfyui rm -rf -- <container-path>`.
- Clash/Mihomo mode switching uses the `external-controller` API.
- Clash Verge start/stop uses configurable local commands.

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
```

## Notes

The terminal implementation is line-oriented with pipes, not a full PTY. It is light and works for ordinary shell commands, downloads, logs, and Docker bash basics. If later you need full-screen terminal programs such as `vim`, `top`, or curses UIs, the next step is adding a PTY backend.
