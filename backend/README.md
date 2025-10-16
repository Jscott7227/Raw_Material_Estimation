# FastAPI Dev Container Starter

This repository ships with a ready-to-go VS Code Dev Container for running and developing the included FastAPI app without touching your local Python toolchain.

## What You Need
- Docker Desktop (or any Docker engine) up and running.
- VS Code with the **Dev Containers** extension (ms-vscode-remote.remote-containers).

## Quick Start
1. **Clone** the repository locally.
2. **Open** the folder in VS Code. When prompted, select *Reopen in Container*. You can also trigger it manually with `> Dev Containers: Reopen in Container`.
3. Wait for the build to finish. First start pulls the `python:3.11-slim` image, installs the dependencies from `requirements.txt`, and sets up oh-my-bash inside the container.
4. Once the container is ready, the FastAPI app is available. Launch it with:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```
   VS Code will forward the exposed port so you can browse `http://localhost:8000`.

## Live Development Tips
- **Dependencies**: Update `requirements.txt` and run `pip install -r requirements.txt` inside the container. The devcontainer automatically reinstalls requirements on creation.
- **Terminal profile**: oh-my-bash is installed by default, giving you a richer bash prompt inside the container.
- **Hot reload**: `uvicorn` runs with `--reload`, so saving files triggers a restart automatically.
- **Python environment**: The container runs code directly in `/app`, so no local virtualenv is required.

## Troubleshooting
- **Container fails to start**: Make sure Docker has at least 4 GB of memory and that there are no conflicting containers using port 8000.
- **Extensions missing**: Install any language tooling (formatters, linters) from within the dev container so they persist in the container file system.
- **Reset the container**: Run `> Dev Containers: Rebuild Container` if dependencies or tooling fall out of sync.

## Next Steps
- Add any project-specific docs here (API routes, workflows, etc.).
- Update `.devcontainer/devcontainer.json` if you need additional tools or different lifecycle commands.
