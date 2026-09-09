# PRD (Final) — Multilingual Sentiment & Emotion Analysis
## Free Stack · Full-Stack (Frontend + Backend + DB) · Tuned for Local Hardware (i5-1250H / 16GB RAM / GTX 1650 4GB)

---

## 1. Project Summary

A full-stack web app that takes social media text in any supported language and returns:
- **Sentiment**: Positive / Negative / Neutral
- **Emotion distribution**: probability per emotion (Joy, Sadness, Anger, Fear, Surprise, Disgust, Neutral/Others)
- **Detected language**
- Visualized as charts, with history stored in a database, plus a batch-CSV mode.

```
Text: "I finally got my dream job! Can't believe it!"
Sentiment: Positive
Emotion: Joy = 0.91, Surprise = 0.06, Others = 0.03
```

---

## 2. Why Build This

- Sentiment alone is too coarse — "Positive" doesn't tell you if someone is *delighted* or just *relieved*, and those need different responses in a real product.
- Emotion is a stronger signal for urgency/intent (e.g., Anger+Negative on a support ticket vs. Sadness+Negative).
- Social text is multilingual by default — an English-only model misses most of the real world's data.
- It's a substantially stronger project than a binary classifier: multi-task learning + multilingual NLP + a real full-stack app + probabilistic outputs.

---

## 3. Your Hardware & What It Means for Model Choice

**Machine:** Intel i5-1250H, 16GB RAM, NVIDIA GTX 1650 (4GB VRAM), 512GB storage.

This is enough to run the **entire app** (backend, frontend, database, inference) comfortably. The only constraint is **training/fine-tuning** — 4GB VRAM is tight for a full-size transformer. So the model choice below is deliberately picked to **fit and train on your GPU**, not just run inference.

### Model Decision (locked in for this PRD)

| Component | Model | Size | Why it fits your GPU |
|---|---|---|---|
| **Emotion model (you fine-tune this)** | `distilbert-base-multilingual-cased` | ~135M params, 104 languages | Half the size of XLM-R-base; trains on GTX 1650 with batch size 16, seq len 128, fp16, using ~2.5–3GB VRAM. This is your **core deliverable model**. |
| **Sentiment model (pretrained, no training needed)** | `cardiffnlp/twitter-xlm-roberta-base-sentiment` | ~279M params | Only used for **inference**, not training — inference needs far less VRAM/RAM than training, runs fine on your GPU or even CPU (~1–2s/request). |
| **Language detection** | `fastText lid.176.bin` | ~130MB, offline | CPU-only, instant, no GPU needed. |

**Why DistilBERT-multilingual instead of XLM-R-base for the model you train yourself:**
- XLM-R-base full fine-tuning typically needs ~6–8GB VRAM at batch size 16/seq len 128 with fp16 — right at or past your GTX 1650's 4GB limit, high risk of CUDA OOM errors.
- DistilBERT-multilingual-cased is ~40% smaller, trains reliably within 4GB, and still covers 104 languages — a good match for a laptop-scale project.

**If you specifically want XLM-R-base's better accuracy anyway (optional, advanced path):**
Use **LoRA fine-tuning** (via Hugging Face `peft` library) — you freeze the base XLM-R weights and only train small low-rank adapter layers. This shrinks the optimizer's memory footprint dramatically (Adam only tracks states for the tiny adapter params, not all 279M), making it feasible on 4GB VRAM with batch size 8, fp16, gradient accumulation. Slower to set up, better if you want the higher-capacity model. This PRD treats it as a stretch goal (Section 9), not the default path.

### Recommended local training settings (DistilBERT path)
```
batch_size        = 16          # drop to 8 if you hit OOM
max_seq_length     = 128
precision          = fp16 (mixed precision)
gradient_accum_steps = 1        # raise to 2 if you reduce batch_size to 8
optimizer          = AdamW
learning_rate       = 2e-5
epochs              = 4
```
Expect roughly 15–35 minutes per epoch on GoEmotions (43k train rows) on a GTX 1650 — a few hours total for the full run. If that's too slow, do the **first exploratory epoch locally** to confirm the pipeline works, then move the full training run to **free Google Colab (T4, 16GB VRAM)** or **Kaggle** (30 free GPU hrs/week) and bring the checkpoint back to your machine for serving.

---

## 4. Dataset

### Primary dataset: **GoEmotions**
- **Source:** Google Research, 58k Reddit comments, 27 fine-grained emotions + Neutral.
- **Link:** https://huggingface.co/datasets/google-research-datasets/go_emotions
- **License:** Open, free for research/project use.
- **Splits (simplified config):** train = 43,410 · validation = 5,426 · test = 5,427.

### Why GoEmotions and nothing else
One clean, well-documented, widely-used dataset is enough — don't fragment your project across five datasets. Sentiment doesn't need a separate dataset: derive it from the same emotion labels (mapping below), or just use the pretrained Cardiff NLP sentiment model directly (no training needed for sentiment at all).

### Label collapsing (27 → 7 target classes)
GoEmotions' 27 fine-grained labels are too granular for the assignment's example format. Collapse them into 6 core emotions + Neutral:

| Target class | GoEmotions labels folded in |
|---|---|
| `joy` | joy, amusement, excitement, admiration, approval, gratitude, love, optimism, pride, relief |
| `sadness` | sadness, disappointment, grief, remorse |
| `anger` | anger, annoyance, disapproval |
| `fear` | fear, nervousness |
| `surprise` | surprise, realization, curiosity, confusion |
| `disgust` | disgust, embarrassment |
| `neutral` | neutral, desire, caring (or keep as their own if you want finer granularity later) |

Sentiment derived from the same row: `joy → Positive`, `sadness/anger/fear/disgust → Negative`, `surprise/neutral → Neutral` (surprise can go either way — treat as context-dependent, default Neutral unless paired with a positive/negative emotion in the same row).

### Multilingual augmentation (for non-English coverage)
GoEmotions is English-only. To make your fine-tuned model actually multilingual:
1. Take a **stratified sample of ~8,000–12,000 rows** from the collapsed training set (balanced across your 7 classes).
2. Machine-translate them into 2–3 target languages (e.g., Hindi, Spanish, French) using a **free, local** translation model — `Helsinki-NLP/opus-mt-en-hi`, `opus-mt-en-es`, `opus-mt-en-fr` (MarianMT, small ~300MB models, run fine on your GPU or CPU).
3. Concatenate translated rows back into the training set, tagged with a `lang` column, so the final fine-tuning set is genuinely multilingual (English + your chosen languages).

This keeps everything free, local, and within your hardware's capability — no paid translation API needed.

### Dataset folder structure (in your repo)
```
data/
├── raw/
│   └── go_emotions/                  # downloaded via `datasets` library, auto-cached
├── processed/
│   ├── train_collapsed.csv           # id, text, lang, joy, sadness, anger, fear, surprise, disgust, neutral, sentiment
│   ├── val_collapsed.csv
│   ├── test_collapsed.csv
│   ├── train_multilingual_augmented.csv   # original + translated rows
│   └── label_map.json                # class name → index mapping, used by training + inference code
└── scripts/
    ├── download_goemotions.py
    ├── collapse_labels.py
    ├── translate_augment.py
    └── build_splits.py
```

**Processed CSV schema (`train_collapsed.csv`):**
| Column | Type | Description |
|---|---|---|
| `id` | string | original GoEmotions row id |
| `text` | string | cleaned comment text |
| `lang` | string | `en`, `hi`, `es`, `fr` (added during augmentation) |
| `joy`, `sadness`, `anger`, `fear`, `surprise`, `disgust`, `neutral` | int (0/1) | multi-label one-hot, since a comment can carry more than one emotion |
| `sentiment` | string | `Positive` / `Negative` / `Neutral`, derived label for the sentiment head (used only if you also train sentiment yourself instead of using the pretrained model) |

---

## 5. Full Tech Stack (100% Free)

| Layer | Choice |
|---|---|
| ML training | PyTorch + Hugging Face `transformers`, `datasets`, `accelerate` (fp16 support) |
| ML serving | Same `transformers` pipeline, loaded once at backend startup |
| Backend | FastAPI (Python) |
| Database | SQLite for local dev (zero setup) → PostgreSQL via Supabase free tier for deployment |
| ORM | SQLAlchemy (async) + Alembic migrations |
| Frontend | React (Vite) + Tailwind CSS + Recharts (or Streamlit for a faster MVP) |
| Language detection | fastText `lid.176.bin` |
| Translation (data augmentation only) | Helsinki-NLP MarianMT models (`opus-mt-en-*`) |
| Experiment tracking (optional) | Weights & Biases free tier |
| Hosting (optional demo) | Frontend → Vercel free tier · Backend → Hugging Face Spaces (free CPU) or Render free tier · DB → Supabase free tier |
| Version control | GitHub |

No paid API keys anywhere — everything runs on open-weight models and free-tier infrastructure.

---

## 6. System Architecture

```
                Social Media Text (raw input)
                           │
                           ▼
                1. Text Preprocessing
   (clean, normalize, emoji-safe, strip URLs/mentions)
                           │
                           ▼
                2. Language Detection
                (fastText lid.176, offline)
                           │
                           ▼
                3. Feature Representation
     (tokenizer of respective model: XLM-R / DistilBERT-multi)
                           │
          ┌────────────────┴────────────────┐
          ▼                                  ▼
  4a. Sentiment Model                4b. Emotion Model
 (pretrained cardiffnlp             (YOUR fine-tuned
  XLM-R, inference only)          DistilBERT-multilingual)
          │                                  │
          └────────────────┬────────────────┘
                           ▼
                5. Result Aggregation
        (sentiment label + emotion vector + confidence)
                           │
                           ▼
                6. Persist to DB + Visualize
             (FastAPI → SQLite/Postgres → React charts)
```

---

## 7. Database Schema

```sql
CREATE TABLE analyses (
    id              SERIAL PRIMARY KEY,
    input_text      TEXT NOT NULL,
    detected_lang   VARCHAR(10),
    sentiment_label VARCHAR(20),
    sentiment_score FLOAT,
    emotion_scores  JSONB,             -- {"joy":0.91,"surprise":0.06,"others":0.03}
    dominant_emotion VARCHAR(30),
    source          VARCHAR(20) DEFAULT 'single',   -- 'single' | 'batch'
    batch_job_id    INTEGER REFERENCES batch_jobs(id) NULL,
    created_at      TIMESTAMP DEFAULT NOW()
);

CREATE TABLE batch_jobs (
    id              SERIAL PRIMARY KEY,
    filename        VARCHAR(255),
    total_items     INTEGER,
    status          VARCHAR(20) DEFAULT 'processing',  -- 'processing' | 'done' | 'failed'
    created_at      TIMESTAMP DEFAULT NOW(),
    completed_at    TIMESTAMP NULL
);
```
Use SQLite locally (`sqlite+aiosqlite:///./app.db`) during development; switch the connection string to Supabase Postgres for deployment — no code changes needed with SQLAlchemy.

---

## 8. API Contract

**`POST /api/analyze`**
Request: `{ "text": "..." }`
Response:
```json
{
  "id": 101,
  "detected_lang": "en",
  "sentiment": { "label": "Positive", "score": 0.97 },
  "emotions": { "joy": 0.91, "surprise": 0.06, "others": 0.03 },
  "dominant_emotion": "joy",
  "created_at": "2026-09-09T10:15:00Z"
}
```

**`POST /api/analyze/batch`** — multipart CSV upload → `{ "batch_job_id": 12, "total_items": 250, "status": "processing" }`

**`GET /api/history?limit=20&offset=0`** — paginated past analyses

**`GET /api/analytics/summary?batch_job_id=12`** — aggregate counts for dashboard:
```json
{
  "sentiment_breakdown": { "Positive": 120, "Negative": 60, "Neutral": 70 },
  "emotion_breakdown": { "joy": 90, "sadness": 40, "anger": 30, "fear": 20, "surprise": 25, "disgust": 15 },
  "top_languages": { "en": 180, "hi": 40, "es": 30 }
}
```

---

## 9. Model Training Recipe (Step-by-Step, GTX 1650-safe)

1. **Load data:** `datasets.load_dataset("google-research-datasets/go_emotions", "simplified")`.
2. **Collapse labels** per Section 4's mapping → save `train_collapsed.csv`, `val_collapsed.csv`, `test_collapsed.csv`.
3. **(Optional) Augment with translated rows** using local MarianMT models → `train_multilingual_augmented.csv`.
4. **Tokenize** with `AutoTokenizer.from_pretrained("distilbert-base-multilingual-cased")`, `max_length=128`, padding/truncation.
5. **Model:** `AutoModelForSequenceClassification.from_pretrained("distilbert-base-multilingual-cased", num_labels=7, problem_type="multi_label_classification")`.
6. **Train** with Hugging Face `Trainer` or a manual PyTorch loop:
   - `fp16=True`, `per_device_train_batch_size=16` (drop to 8 if OOM), `gradient_accumulation_steps=1` (raise to 2 if batch size dropped).
   - `learning_rate=2e-5`, `num_train_epochs=4`, `evaluation_strategy="epoch"`.
7. **Monitor VRAM:** run `nvidia-smi -l 1` in a separate terminal while training; if usage creeps near 3.8–4GB, reduce batch size or sequence length (drop to 96 or 64) before it OOMs.
8. **Evaluate:** Macro-F1, Micro-F1, per-label F1 on the collapsed test set; per-language F1 if you did the multilingual augmentation.
9. **Export:** save the fine-tuned checkpoint locally (`./models/emotion-distilbert-multi/`) and optionally push to your own free Hugging Face Hub repo for easy reloading.
10. **Plug into backend:** point `emotion_pipe = pipeline("text-classification", model="./models/emotion-distilbert-multi", top_k=None)` in `pipeline.py`.

### Stretch: LoRA fine-tuning of XLM-R-base (advanced, optional)
If DistilBERT's accuracy isn't enough and you want XLM-R-base's higher capacity:
```
pip install peft
```
- Load `xlm-roberta-base`, freeze all base weights.
- Wrap with `peft.LoraConfig(r=8, lora_alpha=16, target_modules=["query","value"], lora_dropout=0.1)`.
- Train only the LoRA adapters (a few million params vs. 279M) — this shrinks optimizer memory drastically, making it fit on 4GB with batch size 8, fp16, seq len 128.
- Slower to set up than DistilBERT and not required for a working v1 — treat as a "if I have extra time" improvement.

---

## 10. Project Folder Structure

```
multilingual-sentiment-emotion/
├── data/                          # (see Section 4)
├── notebooks/
│   └── finetune_emotion_model.ipynb
├── models/
│   └── emotion-distilbert-multi/  # your trained checkpoint lives here
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── routers/ (analyze.py, batch.py, history.py, analytics.py)
│   │   ├── ml/ (pipeline.py, preprocess.py)
│   │   ├── db/ (models.py, database.py, crud.py)
│   │   └── schemas.py
│   ├── requirements.txt
│   └── alembic/
├── frontend/
│   ├── src/
│   │   ├── pages/ (Analyzer, Batch, History, HistoryDetail)
│   │   ├── components/ (TextInput, ResultCard, EmotionChart, FileUploader)
│   │   ├── api/client.js
│   │   └── App.jsx
│   ├── package.json
│   └── tailwind.config.js
├── docker-compose.yml
├── README.md
└── PRD.md
```

---

## 11. Frontend Pages

| Page | Purpose |
|---|---|
| `/` Analyzer | Text box → instant sentiment + emotion chart |
| `/batch` | CSV upload → aggregate dashboard |
| `/history` | Table of past analyses, searchable |
| `/history/:id` | Detail view of one past result |

---

## 12. Build Order (Step-by-Step)

1. Set up dataset pipeline: download GoEmotions, collapse labels, build `data/processed/*.csv`.
2. Run a quick baseline training script on your GTX 1650 with a **small subset** (e.g., 2,000 rows, 1 epoch) just to confirm the training loop runs without OOM.
3. Once confirmed, run the **full training job** (locally if time allows, or on free Colab/Kaggle GPU for speed) → save checkpoint to `models/`.
4. Scaffold FastAPI backend: DB models, Alembic migration, `/api/analyze` endpoint wired to your trained emotion model + the pretrained Cardiff sentiment model + fastText language detection.
5. Add `/api/history`, `/api/analytics/summary`, `/api/analyze/batch`.
6. Scaffold React frontend: Analyzer page first, then History, then Batch dashboard.
7. Test end-to-end locally (SQLite).
8. Evaluate your emotion model (Macro-F1, per-class F1, per-language F1) and write up results.
9. (Optional) Deploy: Supabase Postgres + Hugging Face Spaces backend + Vercel frontend.
10. Write README + final report with architecture diagram and metrics.

---

## 13. Evaluation Metrics

| Task | Metric |
|---|---|
| Emotion (multi-label, your fine-tuned model) | Macro-F1, Micro-F1, per-label Precision/Recall, Jaccard similarity |
| Sentiment (pretrained model, evaluated on your labeled subset) | Accuracy, Macro-F1, Confusion Matrix |
| Multilingual robustness | Per-language F1 breakdown (English vs. translated-augmented languages) |
| Qualitative | Manual review of 20–30 hard examples: sarcasm, mixed emotion, code-mixed text |

---

## 14. Testing Checklist

- [ ] Emoji-only text doesn't crash preprocessing
- [ ] Very short text still returns a (low-confidence) result
- [ ] Code-mixed text (e.g., Hinglish) — check language detector confidence and fallback behavior
- [ ] Empty/whitespace input returns a 400, not a crash
- [ ] Large CSV batch (500+ rows) completes without backend timeout
- [ ] GPU training doesn't OOM at the configured batch size — confirmed via `nvidia-smi`
- [ ] DB persists after backend restart (SQLite file / Supabase)

---

## 15. Cost Summary

| Item | Cost |
|---|---|
| GoEmotions dataset | $0 |
| DistilBERT-multilingual / XLM-R base weights | $0 |
| Local training on your GTX 1650 | $0 (just electricity + time) |
| Optional Colab/Kaggle GPU for faster training | $0 |
| FastAPI, SQLAlchemy, React, Tailwind, Recharts | $0 |
| SQLite / Supabase free Postgres | $0 |
| Vercel / Hugging Face Spaces / Render free tiers | $0 |

Total cash cost: **$0**. The only trade-off is training time on your own GPU vs. a free cloud GPU — both are viable, cloud is just faster.
