# Hackathon Dev Container Setup

This project ships with a VS Code Dev Container so contributors can share a consistent development environment. Follow the steps below to get it running locally.

## Prerequisites
- Docker Desktop (macOS/Windows) or Docker Engine (Linux) running and up to date.
- VS Code with the Dev Containers extension (`ms-vscode-remote.remote-containers`).
- Git to clone this repository.

## Quick Start
1. Clone this repository and open the folder in VS Code.
2. When prompted, or via the Command Palette (`⇧⌘P` / `Ctrl+Shift+P`), choose `Dev Containers: Reopen in Container`.
3. Wait for the container build to finish. The build uses `.devcontainer/devcontainer.json`, which points at `docker-compose.yml` and launches the `backend` service with `/app` mounted as the workspace.
4. After the container starts, VS Code runs the configured post-create commands:
   - `pip install -r requirements.txt`
   - `bash -c "$(curl -fsSL https://raw.githubusercontent.com/ohmybash/oh-my-bash/master/tools/install.sh)"` (installs Oh My Bash for a nicer shell experience)
5. Once the commands complete, the environment is ready to use. You can run, test, and debug the app directly inside the container.

## Tips
- Use the Dev Containers status bar menu in VS Code to rebuild the container if you change dependencies or the Docker setup.
- If the post-create commands fail, reopen the Command Palette and run `Dev Containers: Rebuild Container` to retry.
- The dev container installs recommended VS Code extensions automatically; you can add more under `customizations` in `.devcontainer/devcontainer.json`.

