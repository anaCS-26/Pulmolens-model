# PulmoLens Project Context

## Project Overview
PulmoLens is an AI-assisted radiographic diagnostic pipeline that analyzes chest X-rays for 14 pathologies. It combines deep learning (Computer Vision) with Retrieval-Augmented Generation (RAG) to provide visual explainability (heatmaps) and clinical report summaries grounded in official medical guidelines.

### Tech Stack
- **Frontend**: React (TypeScript), Vite, Tailwind CSS. Deployed on **Azure Static Web Apps**.
- **Backend**: FastAPI (Python), Uvicorn. Deployed on **Google Cloud Run**.
- **Deep Learning**: PyTorch (Custom AttentionDenseNet), Grad-CAM++ for explainability.
- **RAG & Agents**: LangChain, Gemma 4 (31B Dense), Pinecone (Vector Store).
- **Persistence & Storage**: Azure Blob Storage (Models/Images), Azure Cosmos DB (Feedback/Audit).
- **Infrastructure**: Azure Bicep, Docker, GitHub Actions CI/CD.

---

## Repository Structure
- `/backend`: FastAPI application, model loading, and RAG logic.
- `/frontend`: React UI componentry and API integration.
- `/ml`: Model architecture definition, training scripts, and evaluation metrics.
- `/infra`: Infrastructure-as-Code (Bicep) and environment configuration.
- `/scripts`: Utility scripts (e.g., Azure cleanup).

---

## Building and Running

### Backend
1. **Environment**: Create a `.env` in the `backend/` directory based on `.env.example`.
2. **Install Dependencies**:
   ```powershell
   cd backend
   pip install -r requirements.txt
   ```
3. **Run Locally**:
   ```powershell
   uvicorn api:app --reload
   ```
4. **Docker Build**:
   ```powershell
   docker build -t pulmolens-backend ./backend
   ```

### Frontend
1. **Environment**: Create a `.env` in the `frontend/` directory based on `.env.example`.
2. **Install Dependencies**:
   ```powershell
   cd frontend
   npm install
   ```
3. **Run Locally**:
   ```powershell
   npm run dev
   ```
4. **Build for Production**:
   ```powershell
   npm run build
   ```

---

## Key Files & Architectural Components

### Backend (`/backend`)
- `api.py`: Main FastAPI entry point. Handles CORS, Prediction, Feedback, and MCP (Model Context Protocol).
- `model.py`: Defines the `AttentionDenseNet` architecture.
- `requirements.txt`: Managed dependencies including PyTorch, LangChain, and Azure SDKs.

### Frontend (`/frontend`)
- `src/api.ts`: API client logic for communication with the Cloud Run backend.
- `src/App.tsx`: Main application shell and routing logic.
- `vite.config.ts`: Build configuration (Note: Ensure `VITE_API_BASE_URL` is set via environment).

### Machine Learning (`/ml`)
- `src/models/attention.py` & `densenet.py`: Detailed attention module implementations.
- `models/`: Location for serialized `.pth` model weights (streamed from Azure in production).

---

## Development Conventions

### Backend
- **Port Binding**: The backend server binds to the `PORT` environment variable (required for Google Cloud Run).
- **Model Deserialization**: PyTorch 2.6+ requires `weights_only=False` or specific safe globals for custom architectures; this is toggled via `ALLOW_UNSAFE_MODEL_DESERIALIZATION`.
- **Memory**: Backend requires a minimum of **4GiB RAM** on Cloud Run to load the PyTorch model without OOM.

### Frontend
- **Environment Variables**: Production API URL is managed via `VITE_API_BASE_URL` in `.env.production`.
- **Styling**: Uses Tailwind CSS with a Glassmorphism design system.
- **Vite Configuration**: Do not hardcode the API base URL in `vite.config.ts`; keep it dynamic via environment variables.

  - `azure-static-web-apps-*.yml`: Deploys the frontend to Azure.

---

## Best Coding Practices

### Branching and Workflow
- **No Direct Commits to Main**: Always create a new branch for changes, fixes, or features.
- **Naming Convention**: Use descriptive branch names (e.g., `feat/gemma-integration`, `fix/ui-overlap`).
- **Testing**: Test changes in the development environment or via Azure Static Web App preview environments before merging.
- **Merging**: Once a task is complete and verified, merge the branch into `main` via a Pull Request or git merge.
- **Cleanup**: Delete the temporary branch immediately after merging to keep the repository clean.

