# Cognitwin — Adaptive Cognitive Digital Twin for Personalized Learning

[![Web CI](https://github.com/cognitwin/cognitwin/actions/workflows/web.yml/badge.svg)](https://github.com/cognitwin/cognitwin/actions/workflows/web.yml)
[![Extension CI](https://github.com/cognitwin/cognitwin/actions/workflows/extension.yml/badge.svg)](https://github.com/cognitwin/cognitwin/actions/workflows/extension.yml)
[![API CI](https://github.com/cognitwin/cognitwin/actions/workflows/api.yml/badge.svg)](https://github.com/cognitwin/cognitwin/actions/workflows/api.yml)
[![GNN-KT CI](https://github.com/cognitwin/cognitwin/actions/workflows/gnn.yml/badge.svg)](https://github.com/cognitwin/cognitwin/actions/workflows/gnn.yml)

> **Paper 1 Build Scope**: Dynamic Graph-based Knowledge Tracing (GNN-KT) for estimating concept mastery and forgetting risk using dual-signal telemetry (passive study exposure + active verification micro-checks).

---

## 1. Architecture & Dual-Signal Telemetry

Traditional e-learning platforms treat all learners identically. Cognitwin continuously models a learner's cognitive state as an evolving **Digital Twin Knowledge Graph**:

1. **Passive Study Exposure (`passive_exposure`)**: An opt-in, privacy-preserving Manifest V3 Chrome Extension tracks active focus dwell time on technical documentation, whitepapers, and videos. It attenuates forgetting decay without falsely inflating verified mastery.
2. **Active Verification Micro-Checks (`quiz_attempt`)**: Targeted 2-minute in-app quizzes generate ground-truth labels for our PyTorch Geometric GNN-KT inference engine.
3. **Live Digital Twin**: Real-time React Flow visualization of concept nodes (Mastered, Active, Fading, and At-Risk) with an evaluation panel for researchers (ROC-AUC and RMSE metrics).

```
[Technical Docs / Videos] ---> [Chrome Extension (Active Focus)] 
                                           | (batched /telemetry/passive)
                                           v
[In-App Quick Checks]     ---> [FastAPI Core Backend] <---> [PostgreSQL 16]
 (ground-truth attempts)                   |
                                           v
                               [PyTorch Geometric GNN-KT]
                                           |
                                           v
                            [Live React Flow Digital Twin]
```

---

## 2. Monorepo Structure

```
cognitwin/
├── apps/
│   ├── web/                    # React 18/19, Vite, TypeScript, Tailwind, React Flow
│   └── extension/              # Manifest V3 Chrome Extension (TypeScript, Vite)
├── services/
│   ├── api/                    # FastAPI, SQLAlchemy 2.0 (async), Alembic, PostgreSQL
│   └── gnn_kt/                 # PyTorch Geometric GNN-KT inference & evaluation
├── docs/
│   ├── PROJECT_SPEC.md         # Full technical and telemetry specifications
│   └── DESIGN_SYSTEM.md        # Stitch-aligned academic-modern design tokens
├── .github/workflows/          # Path-filtered CI/CD (web, extension, api, gnn)
├── AGENTS.md                   # Autonomous agent & team operating rules
├── docker-compose.yml          # Containerized PostgreSQL 16
├── .env.example                # Local environment template
└── README.md                   # This quickstart guide
```

---

## 3. Fast Setup (< 5 Minutes)

### Prerequisites
- **Node.js**: v20+ and `npm`
- **Python**: 3.11+
- **Docker & Docker Compose**: (Optional for local PostgreSQL container)

### Step 1: Clone & Configure Environment
```bash
git clone https://github.com/cognitwin/cognitwin.git
cd cognitwin
cp .env.example .env
```

### Step 2: Start PostgreSQL Database
```bash
# Start containerized PostgreSQL 16
docker compose up -d postgres
```
*(Note: If Docker is not available locally, the API automatically falls back to local SQLite for rapid prototyping and health-check verification).*

### Step 3: Install & Start Backend Services
```bash
# Terminal 1: Core API
cd services/api
pip install -e ".[dev]"
uvicorn app.main:app --reload --port 8000

# Terminal 2: GNN-KT Service
cd services/gnn_kt
pip install -e ".[dev]"
python -m pytest
```

### Step 4: Install & Start Frontend Applications
```bash
# Terminal 3: Web Client (React + Vite)
cd apps/web
npm install
npm run dev
# Open http://localhost:5173

# Terminal 4: Chrome Extension Build
cd apps/extension
npm install
npm run build
# Load ./dist folder in chrome://extensions (Developer Mode -> Load unpacked)
```

---

## 4. Google OAuth 2.0 / OIDC Setup

To enable Google Sign-In for local development or production:

1. **Google Cloud Console**:
   - Go to [Google Cloud Console Credentials](https://console.cloud.google.com/apis/credentials).
   - Create a new project (e.g. `Cognitwin Dev`) or select an existing one.
   - Configure the **OAuth Consent Screen** with User Type: **External**, scopes `openid`, `email`, `profile`.
2. **Create OAuth 2.0 Client ID**:
   - Application type: **Web application**.
   - Name: `Cognitwin Web Client`.
   - Authorized JavaScript origins: `http://localhost:5173` (Vite) and `http://localhost:8000` (FastAPI).
   - Authorized redirect URIs: `http://localhost:8000/api/v1/auth/google/callback`.
3. **Configure Environment Variables**:
   - Copy Client ID and Client Secret into your `.env` file:
     ```env
     GOOGLE_CLIENT_ID=your-client-id.apps.googleusercontent.com
     GOOGLE_CLIENT_SECRET=your-client-secret
     GOOGLE_REDIRECT_URI=http://localhost:8000/api/v1/auth/google/callback
     ```
   *(Note: In mock/development environments without Google credentials configured, the OAuth callback endpoint gracefully creates or links mock Google accounts for integration testing).*

---

## 5. Testing & Quality Verification

Run each component's verification suite:

```bash
# Web Client
cd apps/web && npm run lint && npm run typecheck && npm run build

# Chrome Extension
cd apps/extension && npm run lint && npm run typecheck && npm run build

# Core API (FastAPI)
cd services/api && ruff check . && ruff format --check . && pytest

# GNN-KT Service
cd services/gnn_kt && ruff check . && ruff format --check . && pytest
```

---

## 6. Development Governance & Standards

- Adhere strictly to [AGENTS.md](file:///d:/digital%20twin/AGENTS.md) for phase gates, coding conventions, and Paper 2 seam restrictions.
- All UI components must implement tokens from [docs/DESIGN_SYSTEM.md](file:///d:/digital%20twin/docs/DESIGN_SYSTEM.md).
- Complete telemetry payload formats are documented in [docs/PROJECT_SPEC.md](file:///d:/digital%20twin/docs/PROJECT_SPEC.md).
