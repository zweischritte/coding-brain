# Local Ollama Setup (macOS + Linux GPU)

## macOS (host Ollama, fastest)

Prereqs:
- Ollama installed and running on the host (Metal backend).
- Ollama listening on `0.0.0.0:11434` so Docker can reach it.

Set Ollama to listen on all interfaces (one-time):
```
launchctl setenv OLLAMA_HOST 0.0.0.0
pkill -f "ollama serve"
open -a Ollama
```

Verify:
```
lsof -nP -iTCP:11434 -sTCP:LISTEN
```
You should see `ollama` bound to `*:11434`.

Start the stack:
```
docker compose -f openmemory/docker-compose.yml up -d --build
```

Notes:
- The default `OLLAMA_BASE_URL` is `http://host.docker.internal:11434`.
- No extra env vars required on macOS.

## Linux server with NVIDIA GPU

Prereqs:
- NVIDIA drivers + NVIDIA Container Toolkit installed.

Start the stack with GPU override:
```
docker compose --profile docker-ollama -f openmemory/docker-compose.yml -f openmemory/docker-compose.gpu.yml up -d --build
```

Notes:
- GPU override forces `OLLAMA_BASE_URL=http://ollama:11434`.

## Common troubleshooting

- If the app still uses a stale Ollama URL, check the config in the DB and restart.
- You can override at runtime by setting:
  - `OLLAMA_BASE_URL=http://host.docker.internal:11434`
  - `OLLAMA_HOST=host.docker.internal`
