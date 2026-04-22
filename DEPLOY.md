# ComplianceIQ — Deployment Guide

Run these commands **in your terminal** (not here). Everything is already configured.

---

## Step 1 — Push to GitHub

```bash
cd /path/to/FinSight        # your local project folder

git init
git add .
git commit -m "ComplianceIQ — all 6 phases complete (178 tests passing)"
git remote add origin https://github.com/Jainish1019/complianceiq.git
git branch -M main
git push -u origin main
```

> **Create the GitHub repo first** at https://github.com/new  
> Name: `complianceiq` | Private or Public (your choice) | No README (we have one)

---

## Step 2 — Create HuggingFace Space

1. Go to https://huggingface.co/new-space
2. Fill in:
   - **Owner:** Jainish1019
   - **Space name:** complianceiq
   - **SDK:** Docker
   - **Visibility:** Public (when ready) or Private (for now)
3. Click **Create Space**

---

## Step 3 — Set HuggingFace Space Secrets

In your Space → **Settings → Repository secrets**, add these exactly:

| Secret name | Value |
|---|---|
| `POSTGRES_USER` | `YOUR_POSTGRES_USERNAME_HERE` |
| `POSTGRES_PASSWORD` | `YOUR_POSTGRES_PASSWORD_HERE` |
| `POSTGRES_DB` | `YOUR_POSTGRES_DB_HERE` |
| `API_SECRET_KEY` | `YOUR_API_KEY_HERE` |

> These secrets are only for the HF Space. Your local `.env` already has them.

---

## Step 4 — Push to HuggingFace Space

```bash
# Add the HF Space as a remote
git remote add hf https://huggingface.co/spaces/Jainish1019/complianceiq

# Push (you'll be prompted for HF credentials)
# Username: Jainish1019
# Password: your HuggingFace API token (get it at https://huggingface.co/settings/tokens)
git push hf main
```

HF Spaces will automatically build and deploy. Takes ~5-10 minutes on first push.

---

## Step 5 — Run Locally (optional, needs Docker Desktop)

```bash
# First-time setup (~5 min, pulls Ollama models)
make setup

# Start all 11 services
make dev
```

| Service | URL |
|---|---|
| Dashboard | http://localhost:3000 |
| API docs | http://localhost:8081/docs |
| Airflow | http://localhost:8080 (user: admin / Jainish123#) |
| MLflow | http://localhost:5000 |

```bash
# Load sample data (no Airflow needed)
make seed-db

# Run tests
make test
```

---

## HuggingFace Space URL

Your live demo will be at:  
**https://jainish1019-complianceiq.hf.space**

It's already wired into the frontend build and CORS config.
