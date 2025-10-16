# Hackathon Monorepo

Unified workspace for the hackathon project. The repository ships with a Docker-based development workflow that spins up a Python FastAPI backend and a static frontend.

## Prerequisites
- Docker Desktop (macOS/Windows) or Docker Engine (Linux)
- VS Code with the Dev Containers extension (`ms-vscode-remote.remote-containers`)
- Git

## Quick Start (VS Code Dev Container)
1. Clone the repository and open it in VS Code.
2. When prompted, select `Dev Containers: Reopen in Container` (or run it from the Command Palette).
3. The container build uses `docker-compose.yml` to start the `backend` service and mounts the workspace at `/app`.
4. After the container starts, the post-create command runs `pip install -r requirements.txt` to ensure dependencies are installed.
5. The backend is started with Uvicorn in reload mode; exposed port `8000` is forwarded automatically.

## Running Locally with Docker Compose
```bash
docker compose up --build
```
- Backend available at `http://localhost:8000`
- Frontend served from Nginx at `http://localhost:8080`

## Repo Layout
- `backend/` – FastAPI application, Dockerfile, and Python dependencies.
- `frontend/` – Static site compiled into an Nginx container.
- `docker-compose.yml` – Orchestrates backend and frontend for local development.
- `.devcontainer/` – VS Code Dev Container configuration tied to the backend service.

## Troubleshooting
- Rebuild the container if dependencies change: `Dev Containers: Rebuild Container`.
- Ensure Docker Desktop has enough memory (4 GB+ recommended) and that ports `8000` and `8080` are free.
- If the backend dependencies appear missing, run `pip install -r requirements.txt` inside the dev container terminal.
