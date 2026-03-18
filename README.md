# Automated Detection of Papaya Ring Spot Virus Using Image Processing and LLM-Based RAG Framework

## Abstract
This project is a research-grade papaya leaf diagnostic platform designed to detect Papaya Ring Spot Virus using classical image processing, handcrafted feature extraction, SVM-based classification, severity estimation, heatmap-based explainability, and local Retrieval-Augmented Generation for grounded explanation and advisory generation.

The system is designed for research implementation, faculty review, conference/paper demonstration, and future productization.

---

## Core Identity
This is **not** a generic upload-and-classify application.

The system is built as an:

**Image Processing + Handcrafted Feature Engineering + SVM Classification + Severity Analytics + Heatmap Explainability + Local RAG-Based PRSV Explanation Platform**

---

## Features

### Input Modes
- Demo dataset mode
- Single image upload
- Multiple image upload
- ZIP batch upload

### Image Processing Pipeline
- Resize to fixed size
- RGB / Grayscale / HSV conversion
- Denoising
- CLAHE enhancement
- Leaf segmentation
- Symptom enhancement
- Edge analysis
- Chlorosis emphasis

### Feature Extraction
- Brightness
- Green Ratio
- Hue Mean
- Saturation Mean
- Edge Density
- Color Variance
- Entropy

### Classification
- SVM-based Healthy vs Diseased classification
- Heuristic fallback mode if trained model files are unavailable

### Severity Analysis
- Infection percentage
- Severity score
- Severity label
- Reasoning trace

### Explainability
- Edge heatmap
- Severity / symptom heatmap
- Overlay visualizations

### RAG Layer
- Local PRSV knowledge base
- TF-IDF retrieval
- Cosine similarity
- Technical explanation
- Farmer-friendly explanation
- Advisory notes

### Export and Reporting
- Per-image JSON traces
- Batch summary CSV
- Batch summary JSON
- Downloadable ZIP bundle
- Run logs

---

## Folder Structure

```text
prsv_project/
├── app/
├── image_processing/
├── ml/
├── rag/
│   └── kb/
├── data/
│   ├── demo/
│   ├── uploads/
│   ├── extracted/
│   ├── processed/
│   ├── outputs/
│   ├── temp/
│   └── logs/
├── models/
├── reports/
├── scripts/
├── tests/
├── requirements.txt
├── .env.example
├── run.bat
└── README.md