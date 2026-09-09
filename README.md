# MoodMax — Multilingual Sentiment & Emotion Analysis

> Full-stack AI-powered web app that analyzes social media text for **sentiment** (Positive/Negative/Neutral), **emotion distribution** (Joy, Sadness, Anger, Fear, Surprise, Disgust, Neutral), and **language detection** — with charts, history tracking, and batch CSV processing.

## 🧠 Features

- **Multi-task Analysis**: Sentiment classification + emotion distribution in one request
- **Multilingual**: Supports 100+ languages via multilingual transformer models
- **Emotion Radar Chart**: Interactive Recharts visualization of emotion probabilities
- **Batch Mode**: Upload CSV files for bulk analysis with aggregate dashboard
- **History & Search**: Browse, search, and revisit past analyses
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

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Node.js 18+
- NVIDIA GPU (optional, for training; CPU works for inference)

### 1. Backend Setup

```bash
cd backend
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

> [!TIP]
> On Windows, if `python` is not in your PATH or if using an embedded install (e.g. at `C:\Python312\python.exe`), you can directly run:
> ```powershell
> C:\Python312\python.exe -m uvicorn app.main:app --reload --port 8000
> ```


The backend starts with **mock predictions** by default. To use real ML models:

1. Download fastText language model:
   ```bash
   # Place in models/lid.176.bin
   curl -o ../models/lid.176.bin https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin
   ```

2. The sentiment model auto-downloads on first startup (~1.1GB).

3. For the emotion model, train it first (see Training section).

### 2. Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173 in your browser.

### 3. Data Pipeline (for training)

```bash
cd data/scripts
pip install datasets pandas

# Download GoEmotions + collapse labels
python build_splits.py

# With multilingual augmentation (optional, downloads ~900MB)
python build_splits.py --with-translation
```

### 4. Model Training

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