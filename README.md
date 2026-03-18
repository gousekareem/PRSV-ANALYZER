# PRSV Analyzer – Automated Papaya Disease Detection System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-2.x-lightgrey.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-SVM-orange.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-Image%20Processing-green.svg)
![RAG](https://img.shields.io/badge/AI-RAG%20Explanation-purple.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

**An AI-powered web application to detect Papaya Ring Spot Virus (PRSV) from leaf images using Machine Learning, Heatmap Visualization, and Explainable AI.**

</div>

---

## Table of Contents

- [Overview](#-overview)
- [Problem Statement](#-problem-statement)
- [Key Features](#-key-features)
- [Demo](#-demo)
- [System Architecture](#-system-architecture)
- [Technical Pipeline](#-technical-pipeline)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Model Details](#-model-details)
- [RAG Explanation System](#-rag-explanation-system)
- [API Reference](#-api-reference)
- [Results & Output](#-results--output)
- [Future Scope](#-future-scope)
- [Contributing](#-contributing)
- [License](#-license)

---

## Overview

**PRSV Analyzer** is a complete end-to-end intelligent system designed to help farmers and agricultural researchers detect **Papaya Ring Spot Virus (PRSV)** early using a simple leaf photograph. It goes beyond basic classification — it provides:

- **Disease Prediction** → Healthy / PRSV Infected
- **Severity Estimation** → Mild / Moderate / Severe (0–100%)
- **Heatmap Visualization** → Visual map of infected regions
- **AI-Based Explanation** → RAG-powered recommendations for the farmer

> PRSV (Papaya Ring Spot Virus) is one of the most destructive diseases affecting papaya crops worldwide. Early detection is critical to prevent massive crop losses.

---

## Problem Statement

| Challenge | Impact |
|-----------|--------|
| Manual inspection is inaccurate | Misdiagnosis leads to wrong treatment |
| Lab tests (ELISA, PCR) are expensive | Not accessible to small-scale farmers |
| Lab results take days/weeks | Disease spreads rapidly while waiting |
| No visual feedback for farmers | Difficult to understand infection spread |

**Solution:** A fast, low-cost, automated system that any farmer can use with just a smartphone photo.

---

## Key Features

| Feature | Description |
|---------|-------------|
|  **Smart Detection** | SVM-based classifier for Healthy vs PRSV prediction |
|  **Severity Scoring** | Quantifies disease level from 0% to 100% |
|  **Heatmap Overlay** | Color-coded visualization (red = high infection zones) |
|  **RAG Explanation** | Retrieves real PRSV knowledge to generate accurate advice |
|  **Web Interface** | Simple upload-and-get-result browser UI |
|  **Output Storage** | Saves results (image, heatmap, report) per session |
|  **Fast Processing** | Results in seconds, not days |

---

##  Demo

```
User uploads papaya leaf image
        ↓
System preprocesses image (resize, normalize, denoise)
        ↓
Leaf segmentation (removes background)
        ↓
Symptom enhancement (Canny / Sobel / Laplacian)
        ↓
Feature extraction (brightness, entropy, edge density, etc.)
        ↓
SVM predicts: PRSV DETECTED (Confidence: 94%)
        ↓
Severity: 67% → SEVERE
        ↓
Heatmap generated (infected zones highlighted in red)
        ↓
RAG explanation: "Ring-spot patterns detected. Immediate isolation recommended..."
        ↓
Results saved and displayed to user
```

---

##  System Architecture

```
┌─────────────────────────────────────────────────────┐
│                  USER INTERFACE (Web)               │
│         Upload Image → View Results                 │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│              IMAGE PROCESSING MODULE                │
│   Resize → Grayscale → Normalize → Denoise         │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│              LEAF SEGMENTATION MODULE               │
│   HSV Masking → Contour Detection → ROI Extraction  │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│            SYMPTOM ENHANCEMENT MODULE               │
│   Canny Edge → Sobel Filter → Laplacian Transform   │
└──────────────────────┬──────────────────────────────┘
                       │
       ┌───────────────┼───────────────┐
       │               │               │
┌──────▼──────┐  ┌─────▼──────┐  ┌────▼────────┐
│   FEATURE   │  │    SVM     │  │  HEATMAP    │
│ EXTRACTION  │→ │   MODEL    │  │ GENERATOR   │
└─────────────┘  └─────┬──────┘  └────┬────────┘
                       │               │
                 ┌─────▼──────┐        │
                 │  SEVERITY  │        │
                 │ CALCULATOR │        │
                 └─────┬──────┘        │
                       │               │
               ┌───────▼───────────────▼──────┐
               │     RAG EXPLANATION SYSTEM    │
               │  Query → Retrieve → Generate  │
               └───────────────┬──────────────┘
                               │
                   ┌───────────▼──────────┐
                   │    OUTPUT DISPLAY    │
                   │  Prediction + Score  │
                   │  Heatmap + Advice    │
                   └──────────────────────┘
```

---

##  Technical Pipeline

### Step 1 — Image Preprocessing
Prepares raw image for analysis.
- Resize to standard dimensions (e.g., 224×224)
- Convert RGB → Grayscale
- Normalize pixel values to [0, 1]
- Apply Gaussian blur to remove noise

### Step 2 — Leaf Segmentation
Removes background, isolates the leaf region.
- Convert image to HSV color space
- Create mask for green color range
- Apply morphological operations
- Extract largest contour (the leaf)

### Step 3 — Symptom Enhancement
Highlights disease-related patterns in the image.

| Technique | Detects |
|-----------|---------|
| Canny Edge Detection | Ring-spot boundary patterns |
| Sobel Filter | Edge gradients (lesion borders) |
| Laplacian Transform | Texture irregularities |

### Step 4 — Feature Extraction
Converts visual symptoms into numerical data for ML.

| Feature | Description | Disease Relevance |
|---------|-------------|-------------------|
| Brightness | Mean pixel intensity | Chlorosis detection |
| Green Ratio | Proportion of green pixels | Healthy leaf indicator |
| Entropy | Texture complexity (Shannon) | Pattern irregularity |
| Edge Density | Density of detected edges | Disease spot intensity |
| Laplacian Variance | Image sharpness/roughness | Surface texture changes |

### Step 5 — SVM Classification
Classifies the leaf as Healthy or PRSV-infected.
- **Model:** Support Vector Machine (SVM)
- **Kernel:** RBF (Radial Basis Function)
- **Input:** Feature vector (5 features)
- **Output:** Class label + Confidence score

**Why SVM?**
- Works well on small to medium datasets
- Effective in high-dimensional feature spaces
- Robust against overfitting with proper regularization
- Proven performance in agricultural disease classification

### Step 6 — Severity Estimation
Quantifies how advanced the infection is.

**Formula:**
```
severity = weighted_combination(
    edge_density,
    entropy,
    (1 - green_ratio),
    laplacian_variance
)
```

| Severity Range | Level | Recommended Action |
|---------------|-------|-------------------|
| 0 – 25% | 🟢 Mild | Monitor plant, preventive spray |
| 25 – 50% | 🟡 Moderate | Apply targeted treatment |
| 50 – 100% | 🔴 Severe | Isolate plant, consult expert |

### Step 7 — Heatmap Generation
Produces a visual overlay showing infected zones.
- Combines edge density map + gradient magnitude
- Applies colormap (cool-to-warm: blue → green → red)
- Overlays on original leaf image
- **Red zones = highest infection concentration**

### Step 8 — RAG Explanation System
Generates human-readable explanations using Retrieval-Augmented Generation.

```
Detected Symptoms
      ↓
Build Query String
      ↓
Search PRSV Knowledge Base (vector/keyword search)
      ↓
Retrieve Relevant Passages
      ↓
Generate Contextual Explanation
      ↓
Include Treatment Recommendations
```

---

##  Project Structure

```
prsv_project/
│
├── app.py                      # Main Flask application entry point
│
├── static/
│   ├── css/
│   │   └── style.css           # Stylesheet for web UI
│   ├── js/
│   │   └── main.js             # Frontend interactivity
│   └── images/
│       └── logo.png            # App logo/branding
│
├── templates/
│   ├── index.html              # Upload page
│   ├── result.html             # Result display page
│   └── about.html              # About / project info page
│
├── models/
│   ├── svm_model.pkl           # Trained SVM classifier
│   └── scaler.pkl              # Feature scaler (StandardScaler)
│
├── utils/
│   ├── preprocessing.py        # Image resize, normalize, denoise
│   ├── segmentation.py         # Leaf segmentation (HSV masking)
│   ├── feature_extraction.py   # Feature engineering functions
│   ├── severity.py             # Severity calculation logic
│   └── heatmap.py              # Heatmap generation functions
│
├── rag/
│   ├── knowledge_base.txt      # PRSV disease knowledge corpus
│   ├── rag_engine.py           # RAG retrieval and generation logic
│   └── embeddings/             # (Optional) vector embeddings cache
│
├── outputs/
│   └── [session_id]/
│       ├── original.jpg        # Input image copy
│       ├── heatmap.jpg         # Generated heatmap
│       └── report.json         # Prediction + severity + explanation
│
├── training/
│   ├── train_model.py          # Script to train the SVM model
│   ├── dataset/
│   │   ├── healthy/            # Healthy papaya leaf images
│   │   └── prsv/               # PRSV-infected leaf images
│   └── evaluate.py             # Model evaluation and metrics
│
├── requirements.txt            # Python dependencies
├── config.py                   # App configuration (paths, thresholds)
└── README.md                   # This file
```

---

##  Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- (Optional) Virtual environment tool

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/prsv-analyzer.git
cd prsv-analyzer
```

### 2. Create Virtual Environment (Recommended)
```bash
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on macOS / Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Verify Model Files
Make sure these files exist:
```
models/svm_model.pkl
models/scaler.pkl
```
If not, train the model first (see [Training the Model](#training-the-model)).

### 5. Run the Application
```bash
python app.py
```

Visit: `http://127.0.0.1:5000`

---

##  Requirements

```text
# requirements.txt

flask>=2.0.0
numpy>=1.21.0
opencv-python>=4.5.0
scikit-learn>=1.0.0
scikit-image>=0.18.0
Pillow>=8.0.0
scipy>=1.7.0
matplotlib>=3.4.0
joblib>=1.0.0
```

---

##  Usage

### Web Interface

1. Open browser → `http://127.0.0.1:5000`
2. Click **"Upload Image"**
3. Select a papaya leaf image (JPG, PNG, JPEG)
4. Click **"Analyze"**
5. View results:
   - Disease status (Healthy / PRSV)
   - Confidence score
   - Severity percentage and level
   - Heatmap overlay image
   - AI-generated explanation and recommendations

### Via API (Programmatic)
```bash
curl -X POST http://127.0.0.1:5000/predict \
  -F "image=@path/to/leaf.jpg"
```

**Response:**
```json
{
  "prediction": "PRSV",
  "confidence": 0.94,
  "severity_score": 67.3,
  "severity_level": "Severe",
  "explanation": "Ring-spot patterns with high edge density detected...",
  "recommendation": "Isolate the plant immediately. Apply aphicide...",
  "heatmap_url": "/outputs/abc123/heatmap.jpg"
}
```

---

##  Model Details

### Training the Model

```bash
cd training
python train_model.py
```

This script:
1. Loads images from `training/dataset/healthy/` and `training/dataset/prsv/`
2. Applies preprocessing pipeline
3. Extracts features from all images
4. Trains SVM with cross-validation
5. Saves `models/svm_model.pkl` and `models/scaler.pkl`

### Model Configuration

```python
# config.py
SVM_KERNEL = 'rbf'
SVM_C = 1.0
SVM_GAMMA = 'scale'
FEATURE_SCALER = 'standard'   # StandardScaler
CROSS_VALIDATION_FOLDS = 5
```

### Evaluation Metrics

| Metric | Score |
|--------|-------|
| Accuracy | ~92–95% |
| Precision | ~93% |
| Recall | ~91% |
| F1 Score | ~92% |

> *Metrics may vary based on dataset size and quality.*

---

##  RAG Explanation System

The RAG (Retrieval-Augmented Generation) module provides contextual, knowledge-based explanations instead of hardcoded text.

### How It Works

```python
# rag/rag_engine.py (simplified)

def generate_explanation(prediction, severity, features):
    # Step 1: Build query from analysis results
    query = build_query(prediction, severity, features)

    # Step 2: Search knowledge base
    relevant_passages = search_knowledge_base(query)

    # Step 3: Generate explanation
    explanation = generate_from_context(relevant_passages, prediction, severity)

    return explanation
```

### Knowledge Base
The `rag/knowledge_base.txt` contains curated information about:
- PRSV symptoms and progression stages
- Treatment options (chemical and biological)
- Prevention strategies
- Recommended agronomic practices
- Severity-based response protocols

---

##  Results & Output

Each analysis produces a complete output package:

```
outputs/
└── session_abc123/
    ├── original.jpg         ← Input image (resized)
    ├── heatmap.jpg          ← Color-coded infection map
    └── report.json          ← Full prediction report
```

**Sample Report (report.json):**
```json
{
  "session_id": "abc123",
  "timestamp": "2025-06-15T10:30:00",
  "prediction": "PRSV",
  "confidence": 0.942,
  "severity_score": 67.3,
  "severity_level": "Severe",
  "features": {
    "brightness": 0.421,
    "green_ratio": 0.287,
    "entropy": 4.812,
    "edge_density": 0.563,
    "laplacian_variance": 312.4
  },
  "explanation": "...",
  "recommendation": "..."
}
```

---

##  Impact & Use Cases

| Stakeholder | Benefit |
|------------|---------|
|  Farmers | Early detection → reduced crop loss |
|  Agricultural Universities | Research tool + student projects |
|  Plant Clinics | Rapid screening without lab equipment |
|  NGOs / Aid Orgs | Deploy in low-resource farming regions |
|  AgriTech Companies | Foundation for mobile app integration |

---

##  Future Scope

- [ ] 📱 **Mobile App** (Android / iOS) for field use
- [ ] 📸 **Real-time Camera Detection** (live feed analysis)
- [ ] 🧠 **Deep Learning Upgrade** (CNN / EfficientNet backbone)
- [ ] 🌿 **Multi-Disease Detection** (expand beyond PRSV)
- [ ] 🗣️ **Multilingual Support** (Tamil, Hindi, Telugu for Indian farmers)
- [ ] ☁️ **Cloud Deployment** (AWS / GCP with scalable API)
- [ ] 📊 **Dashboard** (farm-level tracking over time)
- [ ] 🔗 **IoT Integration** (drone or field sensor input)

---

##  Contributing

Contributions are welcome!

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit changes: `git commit -m "Add your feature"`
4. Push to branch: `git push origin feature/your-feature`
5. Open a Pull Request

### Contribution Areas
- Expand the PRSV knowledge base
- Improve SVM feature engineering
- Add deep learning model alternative
- Enhance UI/UX design
- Add multilingual support

---

##  License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

##  Acknowledgements

- OpenCV community for image processing tools
- scikit-learn for the SVM implementation
- Agricultural research papers on PRSV disease patterns
- Flask framework for rapid web development

---

<div align="center">

**⭐ If this project helped you, please give it a star!**

*Built to empower farmers with AI*

</div>
