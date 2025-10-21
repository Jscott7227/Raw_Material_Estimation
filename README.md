# Portobello Raw Materials Demo

FastAPI backend + static dashboard for showcasing live production telemetry at Portobello America. The repo supports both a scripted “one button” demo flow and production‑ready REST endpoints that accept real inventory data.

## Stack
- **Backend**: FastAPI, SQLAlchemy (SQLite for demo); Dockerised with hot reload in dev containers
- **Frontend**: Static HTML/CSS/JS served by Nginx (live reload via bind mount while developing)
- **Tooling**: Docker Compose, VS Code Dev Containers, `load_shipments.py` for bulk imports

---

## Prerequisites
- Docker Desktop (macOS / Windows) or Docker Engine (Linux)
- VS Code with **Dev Containers** extension (`ms-vscode-remote.remote-containers`)
- Git

---

## Quick Start

### Dev Container (recommended)
1. `git clone` the repo and open it in VS Code.
2. Accept the **“Reopen in Container”** prompt (or run *Dev Containers: Reopen in Container* from the Command Palette).
3. The container launch builds the backend image, installs Python deps, and starts FastAPI on port `8000`.
4. The frontend is served by the `frontend` service at `http://localhost:8080` (thanks to the compose bind mount, edits under `frontend/` appear on refresh).

### Docker Compose (host shells)
```bash
docker compose up --build
```
- Backend Swagger UI: http://localhost:8000/docs
- Dashboard: http://localhost:8080

---

## Repository Layout
| Path | Purpose |
|------|---------|
| `backend/` | FastAPI app, ORM models, demo simulators |
| `frontend/` | Static dashboard (HTML/CSS/JS) + Dockerfile |
| `backend/data/shipments.json` | Sample data for `load_shipments.py` |
| `docker-compose.yml` | Orchestrates backend + frontend |
| `.devcontainer/` | VS Code container config pinned to backend service |

---

## How To: Run the Demo
1. **Seed baseline inventory (optional but recommended)**
   ```bash
   curl -X POST http://localhost:8000/api/demo/seed \
        -H 'Content-Type: application/json' \
        -d '{
              "inventory": [
                {"material": "Super Strength 2", "weight": 950},
                {"material": "TN Stone", "weight": 620},
                {"material": "SMS Clay", "weight": 480},
                {"material": "Minspar", "weight": 810},
                {"material": "Sandspar", "weight": 585},
                {"material": "Feldspar", "weight": 720},
                {"material": "LR28", "weight": 540}
              ],
              "reset_deliveries": true,
              "reset_history": true
            }'
   ```
   This sets known starting bins, clears historic deliveries, and primes the inventory history for analytics.

2. **Kick off the one‑button simulation**
   ```bash
   curl -X POST http://localhost:8000/api/simulation/demo
   ```
   Over the next 30 seconds, the backend:
   - generates inbound trucks,
   - consumes inventory based on demo recipe ratios,
   - updates alerts / recommendations.

3. **Open the dashboard** `http://localhost:8080`
   - *Raw Weight* tab shows live bin totals, fill %, reorder warnings, and a 7‑day order outlook.
   - *Truck Data* tab lists deliveries with date range + status filters.

4. **Stop / restart as needed**
   ```bash
   curl -X POST http://localhost:8000/api/simulation/demo/stop
   ```

---

## Working with Real Data
The same backend is ready for production feeds. Replace the demo generators by POSTing real events to the endpoints below:

### Materials (master data + manual adjustments)
- Create: `POST /api/materials`
  ```json
  { "type": "Super Strength 2", "weight": 950, "humidity": 4.5, "density": 1.27 }
  ```
- Update metadata/weight: `PUT /api/materials/{material_id}`
- Apply delta (cycle counts, scrap, etc.): `POST /api/materials/{material_id}/adjust`
  ```json
  { "delta": -12.4, "reason": "Scrap" }
  ```

### Deliveries (TMS / scale integration)
- `POST /api/deliveries`
  ```json
  {
    "delivery_num": "TRK-4581",
    "material_code": "Super Strength 2",
    "incoming_weight_lb": 48200,
    "delivery_time": "2025-10-21T13:40:00",
    "status": "completed"
  }
  ```
  This reuses the Bill of Lading logic: the delivery is stored, inventory is incremented, and a history snapshot is logged for forecasting.

### Analytics & Alerts
- `/api/materials` – enriched material view (bin fill %, reorder flags)
- `/api/alerts` – critical low-bin alerts
- `/api/recommendations?days=7` – projected depletion + order suggestions

Hook these endpoints to your MES/ERP/TMS stack to keep the dashboard live without rerunning the scripted demo.

---

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
