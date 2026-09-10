# MoodMax — Multilingual Sentiment & Emotion Analysis

> Full-stack AI-powered web app that analyzes social media text for **sentiment** (Positive/Negative/Neutral), **emotion distribution** (Joy, Sadness, Anger, Fear, Surprise, Disgust, Neutral), and **language detection** — with charts, batch CSV processing, and aggregate analytics.

## 🧠 Features

- **Multi-task Analysis**: Sentiment classification + emotion distribution in one request
- **Multilingual**: Supports 100+ languages via multilingual transformer models
- **Emotion Radar Chart**: Interactive Recharts visualization of emotion probabilities
- **Batch Mode**: Upload CSV files for bulk analysis with aggregate dashboard
- **Real-time Language Detection**: Automatic language identification via fastText

## 🏗️ Architecture

```
Text Input → Preprocessing → Language Detection (fastText)
                                    ↓
                    ┌───────────────┴───────────────┐
                    ↓                               ↓
           Sentiment Model                   Emotion Model
     (Cardiff NLP XLM-R, pretrained)   (DistilBERT-multi, fine-tuned)
                    ↓                               ↓
                    └───────────────┬───────────────┘
                                    ↓
                        Result Aggregation
                                    ↓
                    Persist to DB → Visualize in React
```

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | FastAPI (Python) |
| Database | SQLite (local) / PostgreSQL (deploy) |
| ORM | SQLAlchemy (async) |
| ML Models | HuggingFace Transformers + PyTorch |
| Emotion Model | DistilBERT-multilingual-cased (fine-tuned) |
| Sentiment Model | cardiffnlp/twitter-xlm-roberta-base-sentiment |
| Language Detection | fastText lid.176.bin |
| Frontend | React + Vite + Vanilla CSS + Recharts |
| Training Data | GoEmotions (Google Research, 58k samples) |

## 📦 Project Structure

```
MoodMax/
├── backend/
│   ├── app/
│   │   ├── main.py           # FastAPI entry point
│   │   ├── schemas.py        # Pydantic models
│   │   ├── routers/          # API endpoints
│   │   ├── ml/               # ML pipeline & preprocessing
│   │   └── db/               # Database models & CRUD
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── pages/            # Analyzer, Batch, History, HistoryDetail
│   │   ├── components/       # Reusable UI components
│   │   ├── api/              # Backend API client
│   │   └── App.jsx           # Router
│   └── package.json
├── data/
│   ├── scripts/              # Data pipeline scripts
│   └── processed/            # Processed datasets
├── training/
│   ├── train.py              # Model training script
│   └── evaluate.py           # Model evaluation script
├── models/                   # Trained model checkpoints
├── docker-compose.yml
└── PRD.md
```

## 🚀 How to Run the Project

### Prerequisites
- **Python 3.10+** (with `pip`)
- **Node.js 18+** & **npm**
- *(Optional)* **Docker & Docker Compose** (for containerized run)
- *(Optional)* NVIDIA GPU (for model training; CPU is sufficient for inference)

---

### Option 1: Run Locally (Two Terminals)

To run the full stack locally, start the backend and frontend in separate terminal windows.

#### 🔹 Terminal 1: Start Backend (FastAPI)

```bash
# 1. Navigate to the backend directory
cd backend

# 2. Create a virtual environment (recommended)
python -m venv venv

# 3. Activate the virtual environment
# Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# Windows (Command Prompt):
venv\Scripts\activate.bat
# Linux / macOS:
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Start the backend server
uvicorn app.main:app --reload --port 8000
```

> [!TIP]
> On Windows, if `python` or `uvicorn` is not in your global PATH, run directly via your Python executable:
> ```powershell
> python -m uvicorn app.main:app --reload --port 8000
> ```

#### 🔹 Terminal 2: Start Frontend (React + Vite)

```bash
# 1. Navigate to the frontend directory
cd frontend

# 2. Install dependencies
npm install

# 3. Start the Vite development server
npm run dev
```

---

### Option 2: Run with Docker Compose

If you have Docker installed, you can start both the backend and frontend with a single command from the project root:

```bash
docker-compose up --build
```

To run in detached background mode:
```bash
docker-compose up -d --build
```

To stop the containers:
```bash
docker-compose down
```

---

### 🌐 Access the Application

Once both services are running, open your browser:

| Service | URL | Description |
|---------|-----|-------------|
| **Frontend Web App** | [http://localhost:5173](http://localhost:5173) | MoodMax React UI |
| **Backend API** | [http://localhost:8000](http://localhost:8000) | FastAPI Base URL |
| **Interactive API Docs (Swagger UI)** | [http://localhost:8000/docs](http://localhost:8000/docs) | Interactive API exploration |
| **ReDoc API Docs** | [http://localhost:8000/redoc](http://localhost:8000/redoc) | Alternative API documentation |
| **Health Check** | [http://localhost:8000/health](http://localhost:8000/health) | Backend status verification |

---

### ⚙️ ML Model Modes

By default, the backend starts in **mock mode** if trained model checkpoints are not found, so you can test the UI and API instantly without downloading gigabytes of weights.

To enable full ML predictions:
1. **Language Detection**: Download fastText model into `models/lid.176.bin`:
   ```bash
   curl -o models/lid.176.bin https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin
   ```
2. **Sentiment Model**: Cardiff NLP XLM-RoBERTa downloads automatically on first run (~1.1GB).
3. **Emotion Model**: Train the custom DistilBERT emotion classifier using the training script (see [Model Training](#-optional-data-pipeline--model-training) below).

---

## 🏋️ Optional: Data Pipeline & Model Training

### 1. Data Pipeline

```bash
cd data/scripts
pip install datasets pandas

# Download GoEmotions + collapse labels
python build_splits.py

# With multilingual augmentation (optional, downloads ~900MB)
python build_splits.py --with-translation
```

### 2. Model Training

```bash
cd training
python train.py --epochs 4 --batch_size 16

# With augmented data
python train.py --use_augmented

# Evaluate
python evaluate.py
```

## 📊 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/analyze` | Analyze single text |
| POST | `/api/analyze/batch` | Upload CSV for batch analysis |
| GET | `/api/history` | Paginated analysis history |
| GET | `/api/history/:id` | Single analysis detail |
| GET | `/api/analytics/summary` | Aggregate analytics dashboard |
| GET | `/health` | Backend health check |

## 💰 Cost

**$0** — Everything runs on open-weight models and free-tier infrastructure.

## 📄 License

MIT