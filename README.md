# Open Human Preference Modeling (OHPM)

## Overview

Open Human Preference Modeling (OHPM) is a comprehensive platform designed to model, predict, and align AI systems with complex human preferences. It integrates multi-modal signal processing (EEG, physiological), advanced reinforcement learning techniques (SFT, DPO), active learning strategies, and privacy-preserving federated inference into a unified, production-grade system.

The platform is architected as a set of decoupled microservices, enabling scalable data ingestion, model training, and real-time inference.

## System Architecture

The system is composed of ten core components:

1.  **User State Encoder & Dynamic Memory System:** A Transformer-based sequence model that encodes user interaction history into latent state embeddings, augmented by a retrieval-augmented generation (RAG) memory system using Pinecone vector storage.
2.  **EEG/Physiological Signal Encoder:** A real-time signal processing pipeline that aligns EEG (OpenBCI) and physiological (Empatica E4) signals with textual modalities using contrastive learning.
3.  **Supervised Fine-Tuning (SFT) Pipeline:** An orchestration engine for generating synthetic preference data and fine-tuning Large Language Models (LLMs) using LoRA adapters.
4.  **Direct Preference Optimization (DPO):** An end-to-end training system for aligning models with human preferences using on-policy generation and iterative refinement.
5.  **Active Learning Query Selector:** An intelligent sample selection module utilizing Inverse Information Density (IID) and uncertainty sampling to optimize annotation budgets.
6.  **Calibration & Bias Detection:** A monitoring system implementing Expected Calibration Error (ECE) metrics, drift detection, and automated fairness correction.
7.  **Human-in-the-Loop Annotation Interface:** A React-based workspace for collecting high-quality human feedback via pairwise comparisons, ranking, and critique tasks.
8.  **Evaluation & Monitoring Dashboard:** A comprehensive observability platform providing real-time insights into system health, model performance, and business KPIs.
9.  **Privacy-Preserving Inference Engine:** A federated learning infrastructure enabling on-device inference and secure gradient aggregation to protect user data.
10. **End-to-End Integration:** A production-ready deployment suite using Kubernetes, Istio, and Terraform for robust operations and continuous delivery.

## Technology Stack

-   **Backend:** Python 3.9+, FastAPI, PyTorch, Transformers, scipy, mne, pylsl
-   **Frontend:** TypeScript, React, Vite, Zustand, TanStack Query, Three.js, D3/Visx
-   **Data & Storage:** PostgreSQL, Redis, Pinecone, TimescaleDB, Elasticsearch
-   **Infrastructure:** Docker, Kubernetes, Helm, Terraform, Prometheus, Grafana, Jaeger

## Installation

### Prerequisites

-   Python 3.9 or higher
-   Node.js 20 or higher
-   Docker and Docker Compose (optional, for full stack deployment)

### Backend Setup

1.  Navigate to the project root.
2.  Create and activate a virtual environment.
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    ```
3.  Install dependencies.
    ```bash
    pip install -r requirements.txt
    ```

### Frontend Setup

1.  Navigate to the frontend directory.
    ```bash
    cd frontend
    ```
2.  Install dependencies.
    ```bash
    npm install
    # or
    pnpm install
    ```

## Usage

### Running the Backend

To start the unified backend API server:

```bash
./start_backend.sh
```

The API will be available at `http://localhost:8000`, with interactive documentation at `http://localhost:8000/docs`.

### Running the Frontend

To start the frontend development server:

```bash
cd frontend
npm run dev
```

The application will be accessible at `http://localhost:5173`.

## Testing

The project includes a comprehensive test suite covering unit, integration, and end-to-end scenarios.

To run backend tests:

```bash
pytest
```

To run frontend tests:

```bash
cd frontend
npm test
```

## License

Copyright (c) 2024 Open Human Preference Modeling. All rights reserved.
