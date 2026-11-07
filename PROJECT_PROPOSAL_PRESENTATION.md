---
marp: true
theme: default
paginate: true
header: "MoodMax — Project Kickoff Presentation"
footer: "Project Proposal · Before Starting Development"
style: |
  section {
    font-family: 'Segoe UI', Arial, sans-serif;
    background: #0f172a;
    color: #f8fafc;
    padding: 36px 48px;
    font-size: 24px;
  }
  h1 {
    color: #38bdf8;
    font-size: 44px;
    margin-bottom: 12px;
  }
  h2 {
    color: #818cf8;
    font-size: 34px;
    border-bottom: 2px solid #334155;
    padding-bottom: 8px;
    margin-top: 0;
  }
  h3 {
    color: #a5b4fc;
    font-size: 26px;
    margin-bottom: 8px;
  }
  strong {
    color: #38bdf8;
  }
  code {
    background: #1e293b;
    color: #f472b6;
    padding: 2px 6px;
    border-radius: 4px;
    font-size: 0.9em;
  }
  ul {
    margin-top: 10px;
    line-height: 1.5;
  }
  li {
    margin-bottom: 10px;
  }
  table {
    font-size: 0.78em;
    width: 100%;
    border-collapse: collapse;
    margin-top: 10px;
  }
  th {
    background: #1e293b;
    color: #38bdf8;
  }
  td, th {
    border: 1px solid #334155;
    padding: 8px 12px;
  }
  blockquote {
    background: #1e293b;
    border-left: 4px solid #38bdf8;
    padding: 8px 14px;
    font-size: 0.78em;
    color: #94a3b8;
    margin-top: 14px;
  }
---

# MoodMax
## Multilingual Sentiment & Emotion Analysis
### Project Kickoff & Implementation Proposal

* **Stage:** Project Proposal (Ready to Build — Pre-Code Phase)
* **What it is:** An AI web app that detects language, sentiment, and exact emotions from text
* **Target Hardware:** Standard Laptop (Intel i5 · 16GB RAM · GTX 1650 4GB GPU)
* **Total Cost:** **$0.00** (100% Free & Open-Source)

> **Speaker Notes:**
> "Hello everyone! Today, we are presenting our project proposal for MoodMax. We have completed all the research, chosen our datasets, designed our system, and planned our hardware limits. We haven't written the code yet — today we are showing you exactly what we are going to build and how we will do it."

---

<!-- SLIDE 2: AGENDA -->
## Agenda: What We Will Cover Today

1. **The Problem:** Why simple "Positive/Negative" is not enough
2. **The Solution:** What MoodMax will do
3. **Hardware Feasibility:** Why our model will run on a normal 4GB GPU
4. **The Dataset:** Google GoEmotions & organizing 27 emotions into 7
5. **Going Multilingual:** Teaching the AI to understand non-English text
6. **Model Selection:** Why we picked DistilBERT Multilingual
7. **System Architecture:** How the frontend, backend, and AI work together
8. **The Web Experience:** Live demo idea & batch file upload
9. **Implementation Plan:** Our 5-step roadmap to build this
10. **Questions & Discussion**

> **Speaker Notes:**
> "Here is our roadmap for this presentation. We will explain the problem with current sentiment tools, introduce our solution, explain our AI and data strategy, walk through the system design, and share our step-by-step plan to build it."

---

<!-- SLIDE 3: THE PROBLEM -->
## The Problem: Current Sentiment Tools are Too Simple

### 1. "Positive" or "Negative" misses the real feeling
* If a customer writes: *"Your product broke on day one!"* $\rightarrow$ That is **Anger**.
* If a customer writes: *"I am so sad this did not work for me"* $\rightarrow$ That is **Sadness**.
* Both are "Negative", but an angry customer needs urgent help, while a sad customer needs gentle support.

### 2. The internet is not English-only
* Most people on social media write in their own languages (Hindi, Spanish, French, or mixed "Hinglish").
* Most basic AI projects only work in English and fail on real-world text.

### 3. Humans feel more than one emotion at once
* You can be both **Happy** and **Surprised** at the same time!

> **Speaker Notes:**
> "Why are we building this? Because traditional sentiment analysis is too basic. Just knowing a review is 'Negative' doesn't tell a company if the user is furious or just disappointed. Also, most global social media comments are not in English. We need a system that understands exact emotions and multiple languages."

---

<!-- SLIDE 4: THE SOLUTION -->
## The Solution: MoodMax

### An intelligent full-stack web app that gives deep text insights:

* **Step 1: Detects Language** automatically (supports 100+ languages).
* **Step 2: Identifies Sentiment** (Positive, Negative, or Neutral).
* **Step 3: Measures 7 Core Emotions** with exact percentage scores:
  * **Joy · Sadness · Anger · Fear · Surprise · Disgust · Neutral**
* **Step 4: Beautiful Visuals** with color bars and radar charts.

```
Example Input: "I finally got the job offer! I still can't believe it! 🎉"
Language: English
Sentiment: Positive (96%)
Emotions: Joy: 91% | Surprise: 7% | Neutral: 2%
Dominant Emotion: Joy
```

> **Speaker Notes:**
> "Here is our solution: MoodMax. You give it any sentence from Twitter, Reddit, or customer feedback. It automatically detects the language, determines the sentiment, and gives you a breakdown of the 7 core human emotions. It turns messy text into clear, actionable charts."

---

<!-- SLIDE 5: HARDWARE CONSTRAINT -->
## Our Hardware: Making AI Work on a Normal Laptop

### The Challenge:
* Big AI models usually need expensive cloud servers with 16GB–24GB of GPU memory.
* Our machine is a regular laptop:
  * **CPU:** Intel Core i5-1250H
  * **RAM:** 16GB
  * **GPU:** NVIDIA GeForce GTX 1650 with **only 4GB VRAM**

### Our Smart Plan:
* We deliberately picked lightweight, efficient models.
* Our chosen model uses **less than 3GB of GPU memory** during training.
* **Result:** We can build, train, and run everything locally on our laptop without crashing or spending money on cloud GPUs!

> **Speaker Notes:**
> "A key strength of our project proposal is practical engineering. We are using a laptop with a GTX 1650 4GB GPU. Instead of trying to use giant models that run out of memory and crash, we carefully calculated our memory limits. Our entire setup fits comfortably inside 3GB of VRAM."

---

<!-- SLIDE 6: MODEL SELECTION -->
## Choosing the Right AI Model

| Model Candidate | Size | Languages | Memory Needed | Can Our GPU Run It? |
|---|---|---|---|---|
| **XLM-RoBERTa-base** | 279M params | 100 | ~7.2 GB | ❌ No (Crashes with Out Of Memory) |
| **mBERT-base** | 178M params | 104 | ~4.8 GB | ⚠️ Risky (Very close to 4GB limit) |
| **DistilBERT Multilingual** | **135M params** | **104** | **~2.8 GB** | **✅ YES! Perfect fit for our GPU** |

### Why `distilbert-base-multilingual-cased` is the Winner:
* **40% smaller** and **2x faster** than traditional BERT models.
* Retains over **95% of the accuracy**.
* Understands **104 different languages**.
* Safe to train on our 4GB GTX 1650 GPU with zero crashes.

> **Speaker Notes:**
> "This table shows why we picked DistilBERT Multilingual. XLM-RoBERTa needs over 7GB of memory, which our laptop doesn't have. DistilBERT has 135 million parameters, supports 104 languages, and uses only 2.8GB during training. It's fast, accurate, and perfectly fits our hardware."

---

<!-- SLIDE 7: THE DATASET -->
## Our Training Data: Google GoEmotions

### Where the data comes from:
* Created by **Google Research**.
* Contains **58,000 real comments** from Reddit.
* Already divided into standard splits:
  * **Train:** 43,410 comments
  * **Validation:** 5,426 comments
  * **Test:** 5,427 comments

### Why this dataset is great:
* It has real human internet language: slangs, emojis, informal typing, and mixed feelings.
* High quality: every comment was labeled by multiple human reviewers.
* Free and open for research and project use.

> **Speaker Notes:**
> "For training our emotion model, we are using Google's GoEmotions dataset. It contains 58,000 real comments from Reddit. This is great because it has real conversational language, internet slangs, and emojis—exactly what our web app will encounter in the real world."

---

<!-- SLIDE 8: SIMPLIFYING THE LABELS -->
## Turning 27 Confusing Emotions into 7 Clean Classes

* Google GoEmotions originally has **27 tiny emotions** (like *grief, remorse, pride, relief*).
* Having 27 emotions is too crowded for users and creates too little data per category.
* **Our Solution:** We group them into the **6 famous Ekman emotions + Neutral**:

| Our 7 Clean Classes | Grouped Google GoEmotions Labels |
|---|---|
| **Joy** | Joy, love, admiration, gratitude, excitement, amusement, pride, relief |
| **Sadness** | Sadness, disappointment, grief, remorse |
| **Anger** | Anger, annoyance, disapproval |
| **Fear** | Fear, nervousness |
| **Surprise** | Surprise, curiosity, confusion, realization |
| **Disgust** | Disgust, embarrassment |
| **Neutral** | Neutral, caring, desire |

> **Speaker Notes:**
> "GoEmotions has 27 separate emotions. Trying to show 27 emotions on a dashboard is confusing and hard to read. We are grouping them into 6 core universal emotions plus Neutral. For example, gratitude and excitement roll into Joy; grief and disappointment roll into Sadness. This gives us clear, reliable results."

---

<!-- SLIDE 9: MULTILINGUAL STRATEGY -->
## Making the AI Multilingual (For $0 Cost)

### The Problem:
* GoEmotions is in English. If we only train in English, the model won't understand other languages well.

### Our 3-Step Free Solution:
1. **Pick 10,000 English comments** from our dataset.
2. **Translate them automatically** into **Hindi, Spanish, and French** using free, offline models (`Helsinki-NLP MarianMT`).
   * No paid API keys needed (like Google Translate or OpenAI).
   * Runs 100% locally on our computer.
3. **Add the translated rows back into our training set**.
* **Result:** Our model learns to detect emotions in English and non-English text equally well!

> **Speaker Notes:**
> "GoEmotions is originally in English. How do we make our model multilingual without paying for translation APIs? We will use free, open-source MarianMT models on our own machine to translate 10,000 comments into Hindi, Spanish, and French. We mix these into the training data, so the model learns cross-lingual emotions naturally."

---

<!-- SLIDE 10: TRAINING RECIPE -->
## How We Will Train the Model

### Our Simple Training Recipe:
* **Batch Size:** 16 (or 8 if memory gets tight)
* **Text Length:** 128 words (covers 98% of social media comments)
* **Speed Boost:** `fp16` mixed precision (cuts memory usage in half)
* **Training Time:** 4 passes (epochs) over the dataset
* **Expected Time:** Around 1 to 2 hours total on our GTX 1650

### Confidence Calibration:
* Neural networks can sometimes be "overconfident" (e.g. saying 99% confident when they're actually unsure).
* We will use **Temperature Scaling** to make sure percentage scores represent honest, reliable probabilities.

> **Speaker Notes:**
> "Here is our training plan. We will use a sequence length of 128 tokens, which fits almost all social comments. Using FP16 mixed precision allows us to train in under 2 hours without running out of memory. We also include temperature calibration so our percentage scores are realistic and trustworthy."

---

<!-- SLIDE 11: SYSTEM ARCHITECTURE -->
## How the Whole System Works

```
      [ User enters text or uploads CSV in Web Browser ]
                             │
                             ▼
      [ FastAPI Backend receives the request ]
                             │
                             ▼
      [ Step 1: Clean text & convert emojis (e.g. 😡 -> angry) ]
                             │
                             ▼
      [ Step 2: Detect language instantly using fastText ]
                             │
                             ▼
      [ Step 3: Run text through our fine-tuned DistilBERT AI ]
                             │
                             ▼
      [ Step 4: Calculate 7 emotion scores & sentiment ]
                             │
                             ▼
      [ Return clean JSON data to the Frontend ]
                             │
                             ▼
      [ Beautiful Interactive Charts displayed in React ]
```

> **Speaker Notes:**
> "Here is the journey of a single request. When a user types a comment, FastAPI cleans the text and converts emojis into words. Then fastText detects the language in less than 1 millisecond. Next, our fine-tuned DistilBERT model scores the 7 emotions. Finally, the backend sends the results back to React to render beautiful charts."

---

<!-- SLIDE 12: BACKEND & API DESIGN -->
## The Backend: Fast and Lightweight

* **FastAPI (Python):** One of the fastest Python web frameworks available.
* **Pre-warmed AI in Memory:**
  * When the backend starts up, the AI model loads once into memory.
  * Every prediction afterwards takes **less than 50 milliseconds**.
* **Zero Database Bottleneck:**
  * Designed to be fast and stateless for live analysis, with optional SQLite storage for saving analysis history.

### Main Endpoints We Will Build:
* `POST /api/analyze` $\rightarrow$ Analyzes a single sentence in real time.
* `POST /api/analyze/batch` $\rightarrow$ Upload a CSV file with hundreds of reviews at once.
* `GET /api/health` $\rightarrow$ Checks if server and model are healthy.

> **Speaker Notes:**
> "For the backend, we chose FastAPI. The model loads into memory once when the server boots up. This means user requests don't have to wait for the model to load every time—each analysis takes less than 50 milliseconds. We will have endpoints for single text analysis and batch CSV uploads."

---

<!-- SLIDE 13: THE API CONTRACT -->
## Simple and Clean API Design

### What the frontend sends:
```json
{
  "text": "Great customer support! Solved my issue in minutes."
}
```

### What the backend returns:
```json
{
  "detected_language": "en (English)",
  "sentiment": { "label": "Positive", "confidence": 0.95 },
  "emotions": {
    "joy": 0.91,
    "surprise": 0.05,
    "neutral": 0.03,
    "sadness": 0.01
  },
  "dominant_emotion": "joy",
  "processing_time": "38ms"
}
```

> **Speaker Notes:**
> "Here is our exact API input and output. The user sends a simple JSON string. The server answers with the detected language, sentiment confidence, exact emotion percentages, the main dominant emotion, and the response time. Everything is clean and simple."

---

<!-- SLIDE 14: FRONTEND EXPERIENCE -->
## The Frontend: Modern, Responsive & Visual

* Built using **React + Vite** for instant loading.
* Styled with **custom modern dark-mode CSS** (no clunky defaults).

### 2 Key Features in the App:
1. **Interactive Text Box:**
   * Type or paste any text in any language.
   * Instant color-coded badges (Green for Positive, Red for Negative, Yellow for Neutral).
   * **Radar Chart & Progress Bars** showing the exact emotion percentages.
2. **Batch CSV Upload:**
   * Drop in a CSV file with 500 customer reviews.
   * Get an aggregate summary chart (e.g., 60% Joy, 25% Anger) and download the labeled spreadsheet.

> **Speaker Notes:**
> "The frontend will be built in React and Vite. It will feature two primary views: an interactive live text analyzer with radar charts, and a batch CSV uploader. If a business has a spreadsheet of 500 user reviews, they can drop it in and get full emotion analytics in seconds."

---

<!-- SLIDE 15: TESTING & QUALITY CHECK -->
## How We Will Test the System

### 1. Accuracy Targets:
* **Overall F1 Score:** Target $\ge 60\%$ on our 7 emotion categories.
* High accuracy expected on common classes like **Joy** and **Neutral**.

### 2. Edge Case Tests We Will Run:
* **Emojis Only:** E.g., `"😭💔"` $\rightarrow$ Must detect Sadness without crashing.
* **Mixed Language (Hinglish):** E.g., *"Yeh service bohot bekar hai"* $\rightarrow$ Must detect Negative / Anger.
* **Empty Input:** E.g., `""` $\rightarrow$ Friendly error message, no server crashes.
* **Large Files:** 500+ rows CSV file $\rightarrow$ Must finish smoothly without freezing.

> **Speaker Notes:**
> "How do we make sure MoodMax works well? We have defined clear quality checks. We will test edge cases like emojis, slang, and mixed languages like Hinglish. We will also test error handling so empty text or bad files never crash the system."

---

<!-- SLIDE 16: RISKS & OUR SOLUTIONS -->
## Anticipated Challenges & How We Solve Them

| Potential Challenge | Why it Happens | Our Solution |
|---|---|---|
| **GPU runs out of memory** | 4GB VRAM is tight | Use DistilBERT + mixed precision `fp16` + batch size 16 |
| **Some emotions are rare** | GoEmotions has less "Disgust" than "Joy" | Use class weighting during training so rare emotions aren't ignored |
| **Short text language detection** | 2-3 word sentences are hard to identify | fastText offline model + fallback to English if confidence is low |
| **Server slows down on big CSVs** | Processing 500 rows takes time | Process in small batches with live progress updates |

> **Speaker Notes:**
> "Every engineering project faces challenges. We have already planned our mitigations: if our GPU gets tight, we drop batch size to 8; if certain emotions are rare in the data, we use weighted loss; and for big CSV files, we process them in streaming batches."

---

<!-- SLIDE 17: ZERO-DOLLAR BUDGET -->
## 100% Free & Open-Source Stack

* **Machine Learning:** PyTorch + Hugging Face Transformers ($0)
* **Dataset:** Google GoEmotions via Hugging Face ($0)
* **Backend:** FastAPI + Python 3.11 ($0)
* **Frontend:** React 19 + Vite + Recharts ($0)
* **Language Detection:** fastText offline model ($0)
* **Translation Augmentation:** Helsinki-NLP MarianMT ($0)
* **Compute:** Local Laptop GTX 1650 GPU ($0)

### Total Cash Cost to Build & Run: **$0.00**

> **Speaker Notes:**
> "One of the best parts about MoodMax is our budget: zero dollars. Every single library, model, dataset, and framework we are using is open-source. We don't need subscriptions, OpenAI tokens, or expensive cloud servers."

---

<!-- SLIDE 18: ROADMAP -->
## Our 5-Step Implementation Roadmap

```
[Sprint 1: Prepare Data]
├── Download GoEmotions dataset
├── Group 27 emotions into 7 core classes
└── Run local translations for Hindi, Spanish, French
         │
         ▼
[Sprint 2: Train the AI Model]
├── Test training loop on 2,000 rows first
├── Run full 4-epoch fine-tuning on our GTX 1650
└── Calibrate confidence scores and verify test accuracy
         │
         ▼
[Sprint 3: Build the FastAPI Backend]
├── Load trained model in memory
├── Add text cleanup, emoji converter & fastText
└── Build /api/analyze and /api/analyze/batch endpoints
         │
         ▼
[Sprint 4: Build the React Frontend]
├── Create modern dark-mode user interface
├── Connect live API with interactive radar charts
└── Add CSV drag-and-drop file uploader
         │
         ▼
[Sprint 5: Testing & Final Polish]
├── Run edge-case tests (emojis, Hinglish, error handling)
└── Package inside Docker for one-click setup
```

> **Speaker Notes:**
> "Here is our 5-step plan. Sprint 1 prepares the data. Sprint 2 trains and validates the AI. Sprint 3 creates the FastAPI backend. Sprint 4 builds the React frontend. And Sprint 5 tests edge cases and packages everything into Docker."

---

<!-- SLIDE 19: WHAT WE WILL DELIVER -->
## What You Will See When We Finish

1. **A Trained Multilingual Emotion Model:**
   * Capable of understanding sentiment and 7 emotions across 100+ languages.
2. **A Full-Stack Interactive Web App:**
   * Type text, see live radar charts, or upload CSV files for bulk analysis.
3. **Sub-50ms Response Speed:**
   * Fast, in-memory architecture with zero lag.
4. **Clean, Documented Codebase:**
   * Modular Python backend, React frontend, and reproducible training scripts.
5. **Complete Project Documentation:**
   * Full metrics report, API documentation, and setup guide.

> **Speaker Notes:**
> "When we finish building, you will have a complete, polished, and working product: a trained multilingual model, an interactive web dashboard, fast APIs, and comprehensive documentation."

---

<!-- SLIDE 20: CONCLUSION -->
## Summary & Ready to Start!

### Why MoodMax will be a success:
* ✅ **Clear Real-World Need:** Emotion analysis is vastly superior to simple positive/negative.
* ✅ **Hardware Feasible:** Engineered specifically to train and run on our 4GB laptop GPU.
* ✅ **Comprehensive Plan:** Dataset, label mapping, model recipe, and UI are fully designed.
* ✅ **$0 Cost:** 100% open-source and ready to build.

---

### Questions & Feedback

**Thank you!**  
*We are ready to start Sprint 1 and begin building MoodMax.*

> **Speaker Notes:**
> "In conclusion: the preparation is complete, the architecture is locked in, and our hardware limits are respected. We have everything we need to start building MoodMax. Thank you for your time, and we would love to answer any questions or hear your feedback!"
