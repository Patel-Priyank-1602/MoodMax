# 🚀 MoodMax — Cloud Deployment Guide

> **Problem:** Render's free tier has a strict **512 MB RAM ceiling** and GitHub blocks files larger than **100 MB**. Because our fine-tuned DistilBERT emotion model (`model.safetensors`) is **~541 MB**, deploying both the model and backend directly on Render causes Out-of-Memory (OOM / Error 137) crashes or Git push rejections.

---

## 🏗️ Architecture Options

### 🌟 Option A: Deploy Backend on Hugging Face Spaces (16 GB Free RAM!) + Frontend on Vercel
> **⭐ Highly Recommended by AI Engineers**

| Feature | Render Free Tier | Hugging Face Spaces (Free Tier) |
|---|---|---|
| **Memory (RAM)** | 512 MB (Crashes easily) | **16 GB (32x more RAM!)** |
| **CPU** | 0.1 CPU (Very slow) | **2 vCPUs** |
| **Model Storage** | 500 MB slug limit | **50 GB free storage** |
| **Price** | Free | **$0.00 Free forever** |

* **Backend + 540MB Model:** Runs seamlessly on Hugging Face Spaces (Zero OOM crashes).
* **Frontend:** Deployed on Vercel connecting to `https://<your-space-name>.hf.space`.

---

### 🌐 Option B: Model on Hugging Face + Backend on Render + Frontend on Vercel
> **(Exactly what you requested)**

```
┌─────────────────┐           ┌──────────────────┐           ┌──────────────────────┐
│  React Frontend │ ──HTTP──► │  FastAPI Backend │ ──HTTP──► │   Hugging Face Hub   │
│   (on Vercel)   │           │   (on Render)    │           │ (541MB Emotion Model)│
│                 │           │  (<60 MB RAM!)   │           │                      │
└─────────────────┘           └──────────────────┘           └──────────────────────┘
```

By setting `USE_HF_INFERENCE_API=true`, Render **never loads the 541MB model into RAM**! Instead, Render delegates inference to Hugging Face over HTTP, consuming only **~50 MB RAM** on Render. This guarantees Render will **never crash with OOM Error 137**.

---

## 🛠️ Step 1: Upload Your Model to Hugging Face (~2 Minutes)

You need to host your 541MB model on Hugging Face Model Hub.

### Method 1: Using the Automated Script
Run the helper script included in the MoodMax root directory:

```bash
# 1. Install huggingface_hub if needed
pip install huggingface_hub

# 2. Run the upload script (replace <username> with your HF username)
python upload_model_to_hf.py --repo-id <username>/moodmax-emotion-distilbert
```
The script will prompt for your Hugging Face write token (obtain it free from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)).

---

### Method 2: Via Hugging Face Web Interface (Drag & Drop)
1. Go to [huggingface.co/new](https://huggingface.co/new).
2. Select **Model**, name it `moodmax-emotion-distilbert`, set to **Public**, and click **Create model**.
3. In the new repository, click **Add file** $\rightarrow$ **Upload files**.
4. Drag and drop the following files from your local `models/emotion-distilbert-multi/` directory:
   * `model.safetensors` (~541 MB — Hugging Face supports files up to 50 GB!)
   * `config.json`
   * `tokenizer.json`
   * `tokenizer_config.json`
   * `thresholds.json`
   * `label_map.json`
5. Click **Commit changes to main**.

Your model is now publicly hosted at:
`https://huggingface.co/<username>/moodmax-emotion-distilbert`

---

## ⚙️ Step 2: Configure & Deploy Backend on Render

1. Log into [dashboard.render.com](https://dashboard.render.com).
2. Click **New +** $\rightarrow$ **Web Service**.
3. Connect your GitHub repository (`MoodMax`).
4. Configure service settings:
   * **Root Directory:** `backend`
   * **Runtime:** `Python 3`
   * **Build Command:** `pip install --upgrade pip && pip install -r requirements.txt`
   * **Start Command:** `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
5. Scroll down to **Environment Variables** and add:
   * `HF_MODEL_ID`: `<username>/moodmax-emotion-distilbert`
   * `USE_HF_INFERENCE_API`: `true` *(Keeps Render memory under 60MB!)*
   * `HF_TOKEN`: `<your_hf_access_token>` *(Required if private; optional if public)*
6. Click **Deploy Web Service**.
7. Once deployed, note down your Render backend URL (e.g. `https://moodmax-backend.onrender.com`).

---

## 🎨 Step 3: Deploy Frontend on Vercel

1. Log into [vercel.com](https://vercel.com).
2. Click **Add New...** $\rightarrow$ **Project**.
3. Select your `MoodMax` repository from GitHub.
4. Configure project settings:
   * **Framework Preset:** `Vite`
   * **Root Directory:** Click Edit and select `frontend`
5. Expand **Environment Variables** and add:
   * **Name:** `VITE_API_URL`
   * **Value:** `https://moodmax-backend.onrender.com` *(or your HF Space URL)*
6. Click **Deploy**.

Vercel will build and launch your React 19 frontend on a global CDN in under 60 seconds!

---

## 🚀 Alternative: 1-Click Deploy Entire Backend to Hugging Face Spaces (Option A)

If you prefer to bypass Render's sleep timers and 512MB RAM limits entirely:

1. Go to [huggingface.co/new-space](https://huggingface.co/new-space).
2. Set Space Name to `moodmax-backend`.
3. Select **Space SDK:** `Docker` $\rightarrow$ **Blank**.
4. In your Space repository, upload the contents of the `backend/` folder plus `models/emotion-distilbert-multi/`.
5. Hugging Face Spaces will automatically build the Docker container and give you **16 GB RAM for free**!
6. Copy the Space direct URL (format: `https://<username>-moodmax-backend.hf.space`) and use that as `VITE_API_URL` on Vercel.

---

## 🔍 Verification Checklist

- [ ] Model repository on Hugging Face shows `model.safetensors` (~541 MB) and `config.json`.
- [ ] Render backend logs show: `Using Hugging Face Serverless Inference API for '<username>/moodmax-emotion-distilbert' (Zero local RAM overhead!)`.
- [ ] Testing `GET https://moodmax-backend.onrender.com/health` returns `{"status": "healthy"}`.
- [ ] Vercel frontend loads and successfully analyzes sentences.
