# Portobello Raw Materials Demo

Demonstrates automated material pile volume estimation at Portobello America. The system uses AI-powered computer vision to segment material piles, estimate depth, and calculate volumes, integrated into a containerized dashboard with real-time inventory insights.

## Stack
- **Backend**: FastAPI, SQLAlchemy (SQLite for demo), Dockerized with hot reload in dev containers
- **Frontend**: Static HTML/CSS/JS served via Nginx (live reload during development)
- **AI/ML**: Transformer-based image segmentation (Segment Anything Model) + MiDaS depth estimation for material volume calculations
- **Tooling**: Docker Compose, VS Code Dev Containers, `load_shipments.py` for bulk imports

## Quick Start

### Dev Container (recommended)
1. `git clone` the repo and open it in VS Code.
2. Accept the **“Reopen in Container”** prompt. The container builds the backend image, installs dependencies, and starts FastAPI on port `8000`.
3. Dashboard served at `http://localhost:8080`. Edits to `frontend/` appear on refresh thanks to the bind mount.

### Docker Compose (host shell)
```bash
docker compose up --build
```

- **Backend Swagger UI:** [http://localhost:8000/docs](http://localhost:8000/docs)  
- **Dashboard:** [http://localhost:8080](http://localhost:8080)  

## Material Pile Volume Estimation Workflow
1. **Image Segmentation** – The AI model detects material piles in images using the Segment Anything Model (SAM).  
2. **Depth Prediction** – MiDaS estimates the depth map for the segmented regions.  
3. **Volume Calculation** – Using depth and pixel-to-meter conversion, the system computes approximate pile volumes.  
4. **Dashboard Integration** – Volumes, inventory levels, and reorder projections are visualized in real time on the web dashboard.  

> All AI/ML workflows run inside Docker and interact with real REST endpoints for both demo and production-ready data. Sample image inputs and outputs can be found in the `/sample_images` folder.

## Utilities
- Bulk import shipments: `./.venv/bin/python backend/load_shipments.py --file backend/data/shipments.json`
- View API docs: `http://localhost:8000/docs`
- Frontend live reload during development: edits in `frontend/` are served immediately thanks to the Docker volume mount.

---

## Troubleshooting
- **Ports busy**: stop any process listening on `8000`/`8080`.
- **Python deps missing**: inside the dev container run `pip install -r backend/requirements.txt`.
- **Docker resources**: ensure Docker Desktop has at least 4 GB RAM allocated.
- **Reset demo state**: rerun `/api/demo/seed` followed by `/api/simulation/demo`.
