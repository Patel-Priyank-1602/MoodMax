---
language:
- multilingual
- en
- hi
- es
- fr
license: mit
tags:
- emotion-classification
- sentiment-analysis
- distilbert
- goemotions
- text-classification
pipeline_tag: text-classification
widget:
- text: "I finally got the dream job! I can't believe it!"
- text: "The flight was delayed by 4 hours and nobody helped."
- text: "यह सर्विस बिल्कुल बेकार है, मेरा समय बर्बाद हुआ!"
- text: "¡Excelente servicio, muy agradecido por la ayuda!"
---

# 🎭 MoodMax — Multilingual Emotion Classification Model

Fine-tuned `distilbert-base-multilingual-cased` Transformer trained on Google Research's **GoEmotions** dataset (collapsed into 7 standardized Ekman emotion classes) with temperature calibration ($T = 1.1724$) and optimal per-class decision thresholds.

## 📊 Benchmark Metrics
Evaluated on the held-out GoEmotions test split (5,427 comments):
- **Macro-F1:** 60.25%
- **Micro-F1:** 67.29%
- **Macro-Precision:** 57.52%
- **Sigmoid ECE:** 4.54% (Calibrated)

## 🏷️ Supported Emotions (7 Classes)
1. `joy`
2. `sadness`
3. `anger`
4. `fear`
5. `surprise`
6. `disgust`
7. `neutral`

## 🚀 Usage in Python

```python
from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model="Priyank1602/moodmax-emotion-distilbert",
    top_k=None
)

results = classifier("I can't believe we actually won the championship!")
print(results)
```
