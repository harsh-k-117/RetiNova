# RetiNova — Explainable AI for Diabetic Retinopathy Screening

**Smart India Hackathon 2026 · Problem Statement 26038 (MathWorks, MedTech / BioTech / HealthTech)**

| | |
|---|---|
| **Problem Statement ID** | 26038 |
| **Theme** | MedTech / BioTech / HealthTech |
| **Category** | Software |
| **Organization** | MathWorks |
| **Team Name** | RetiNova |

RetiNova is an explainable AI-powered screening system for Diabetic Retinopathy, built entirely in MATLAB. It provides automated quality checks, DR severity grading, and visual explanations to support clinical decision-making in rural healthcare settings.

> **Prototype for educational and research purposes. Clinical validation required for deployment.**

---

## Table of Contents

- [Problem Background](#problem-background)
- [Our Solution](#our-solution)
- [Why This Matters](#why-this-matters)
- [Key Features](#key-features)
- [System Pipeline](#system-pipeline)
- [Tech Stack](#tech-stack)
- [Design Targets](#design-targets)
- [Repository Layout](#repository-layout)
- [Getting Started](#getting-started)
- [Datasets](#datasets)
- [Impact and Benefits](#impact-and-benefits)
- [Future Roadmap](#future-roadmap)
- [Team](#team)
- [Disclaimer](#disclaimer)

---

## Problem Background

India has more than **77 million diabetic adults**, the second highest number globally. Around 18% develop Diabetic Retinopathy (DR), a leading cause of preventable blindness. Early screening can prevent **up to 90% of vision loss**, but India has only about **1 ophthalmologist per 100,000 people** in rural areas.

**The Challenge:**
- Manual screening is impossible at scale
- Existing AI tools are "black boxes" that don't explain their decisions
- Low-quality images from portable cameras break most AI systems
- Delayed diagnosis leads to irreversible vision loss

**RetiNova addresses all these challenges with explainable AI built on MATLAB.**

---

## Our Solution

RetiNova is an end-to-end MATLAB pipeline that:

1. **Validates image quality** before analysis (sharpness, brightness, exposure)
2. **Preprocesses and enhances** fundus images for optimal analysis
3. **Detects retinal structures** (vessels, optic disc, lesions)
4. **Grades DR severity** on the International Clinical DR Scale (0–4)
5. **Explains every prediction** with Grad-CAM visual heatmaps
6. **Provides referral recommendations** based on clinical thresholds
7. **Simulates telemedicine workflows** to help health programs plan resources

---

## Why This Matters

| Problem | How RetiNova Helps |
|---|---|
| Very few ophthalmologists in rural India | AI performs first-level screening; doctors review only flagged cases |
| Poor image quality from portable cameras | Built-in quality validation and enhancement pipeline |
| Black-box AI reduces clinician trust | Grad-CAM shows exactly where abnormalities were detected |
| Delayed diagnosis causes vision loss | Faster triage enables earlier treatment and referral |
| Limited healthcare planning data | Simulink workflow models estimate capacity and resources |

---

## Key Features

### 🔍 **Intelligent Image Quality Assessment**
- Automatic validation of sharpness, brightness, contrast, and exposure
- Rejects ungradable images and requests recapture
- Calibrated thresholds based on real clinical datasets

### 🧠 **AI-Powered DR Grading**
- 5-level classification (Grade 0–4) using International Clinical DR Scale
- Ensemble model architecture for robust predictions
- Trained on IDRiD dataset with proper validation methodology

### 💡 **Explainable AI with Grad-CAM**
- Visual heatmaps highlight regions influencing predictions
- Helps clinicians verify AI reasoning in under 30 seconds
- Interactive opacity slider for overlay adjustment

### 🩺 **Clinical Decision Support**
- Clear referral flags for Grade ≥ 2 (referable DR)
- Probability scores across all severity levels
- Confidence metrics for each prediction

### 🖥️ **Interactive Review Application**
- Clean MATLAB UI for clinician workflow
- Side-by-side original image and Grad-CAM overlay
- Real-time quality assessment and grading

### 🏥 **Telemedicine Workflow Simulation**
- Simulink model for screening workflow optimization
- Helps district health programs plan bandwidth and staffing
- Estimates processing time and ophthalmologist workload

---

## System Pipeline

The complete screening workflow from image capture to clinical decision:

```
┌─────────────────────┐
│  Fundus Image       │
│  (Portable Camera)  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Quality Check      │◄─── Fails? Request Recapture
└──────────┬──────────┘
           │ Passes
           ▼
┌─────────────────────┐
│  Image Enhancement  │
│  (CLAHE, Denoise,   │
│   Illumination Fix) │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Retinal Analysis   │
│  • Vessels          │
│  • Optic Disc       │
│  • Lesions          │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  DR Grading (0-4)   │
│  + Grad-CAM         │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Clinician Review   │
│  (<30 seconds)      │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Referral Decision  │
└─────────────────────┘
```

---

## Tech Stack

| Component | Technology |
|---|---|
| **Core Platform** | MATLAB (R2023a+) |
| **Image Processing** | Image Processing Toolbox |
| **Computer Vision** | Computer Vision Toolbox |
| **Deep Learning** | Deep Learning Toolbox, ONNX Model Support |
| **Medical Imaging** | Medical Imaging Toolbox |
| **Statistics** | Statistics and Machine Learning Toolbox |
| **Workflow Simulation** | Simulink / SimEvents |
| **Explainability** | Grad-CAM (Gradient-weighted Class Activation Mapping) |
| **Hardware** | CUDA-capable GPU for training (inference works on CPU) |

---

## Design Targets

Performance goals aligned with clinical requirements:

| Metric | Target | Notes |
|---|---|---|
| **Sensitivity** | > 90% | For referable DR detection (Grade ≥ 2) |
| **Specificity** | > 85% | Minimize unnecessary referrals |
| **Review Time** | < 30 seconds | Per image with Grad-CAM |
| **Quality Rejection** | Automatic | Prevent ungradable image analysis |

---

## Repository Layout

```
RetiNova/
├── drGradingPrototype.m          # Main UI application
├── predictIDRiDGrader.m          # Inference with the fold ensemble + Grad-CAM
├── trainIDRiDGrader.m            # 3-fold cross-validated training and out-of-fold report
├── assessQuality.m               # Image quality module
├── preprocessIDRiDImage.m        # Image preprocessing used for caching / inference
├── blendCamOverlay.m             # Heatmap blending for the app
├── computeGradCAM.m              # Grad-CAM computation
├── lesionSaliencyMap.m           # Heuristic lesion heatmap
├── loadDemoGradingNetwork.m      # Demo network loader (when models missing)
│
├── segmentation/                 # Segmentation module
│   ├── segmentVessels.m          # Vessel segmentation
│   └── loadIDRiDSegmentation.m   # Lesion mask loader
│
├── test/                         # Development & testing scripts
│   ├── main.m                    # Alternative training script (older version)
│   ├── demoGradeScores.m         # Placeholder scoring (for demo mode)
│   ├── loadIDRiDGradingData.m    # Data loader utility
│   ├── test.m                    # Manual testing
│   ├── demo.m                    # Quality analysis
│   ├── evaluateVesselsDrive.m    # Vessel evaluation
│   ├── evaluateDrDetectOnIdrid.m # ONNX testing
│   ├── testDrDetectOnnx.m        # ONNX import test
│   └── evaluateIDRiDGrader.m     # Model evaluation
│
├── models/                       # Trained models (generated)
│   └── idrid_grade5/             # Training artifacts
│
├── data/                         # Datasets (download separately)
│   ├── B. Disease Grading/       # IDRiD grading
│   └── A. Segmentation/          # IDRiD segmentation
│
├── +efficientnet_b0_regression_512px/  # Auto-generated MATLAB code
├── README.md
├── MATLAB_GRADING.md
├── SIH-Pitch-deck-RetiNova.pptx
└── Prototype demo video.mp4
```

---

## Getting Started

### Prerequisites

**MATLAB Toolboxes Required:**
- Deep Learning Toolbox
- Image Processing Toolbox
- Computer Vision Toolbox
- Statistics and Machine Learning Toolbox
- Parallel Computing Toolbox (for GPU training)
- Deep Learning Toolbox Converter for ONNX Model Format

**Hardware:**
- CUDA-capable GPU recommended for training
- CPU sufficient for inference/demo

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/harsh-k-117/RetiNova.git
   cd RetiNova
   ```

2. **Open in MATLAB:**
   ```matlab
   cd '/path/to/RetiNova'
   ```

3. **Download datasets** (see [Datasets](#datasets) section below)

### Running the Application

**Launch the interactive UI:**
```matlab
drGradingPrototype
```

**The app allows you to:**
- Load fundus images from disk or use sample images
- Automatically assess image quality
- View DR grade predictions with confidence scores
- Explore Grad-CAM explanations with adjustable opacity
- See referral recommendations

**Training a new model:**
```matlab
trainIDRiDGrader  % 3-fold cross-validation
```

---

## Datasets

RetiNova uses publicly available retinal imaging datasets:

### Primary Dataset

**IDRiD (Indian Diabetic Retinopathy Image Dataset)**
- 516 high-resolution fundus images
- DR grading labels (0–4)
- Lesion segmentation masks
- Source: https://ieeedataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid
- License: Research use (IEEE DataPort terms)

### Additional Validation Datasets

1. **APTOS 2019 Blindness Detection**
   - https://www.kaggle.com/c/aptos2019-blindness-detection
   - Large-scale DR grading dataset

2. **DRIVE (Digital Retinal Images for Vessel Extraction)**
   - https://drive.grand-challenge.org/
   - Vessel segmentation ground truth

3. **Messidor-2**
   - https://www.adcis.net/en/third-party/messidor2/
   - External validation dataset

### Data Setup

Download IDRiD and extract to `data/` folder:

```
data/B. Disease Grading/B. Disease Grading/
    1. Original Images/
        a. Training Set/   (413 images)
        b. Testing Set/    (103 images)
    2. Groundtruths/
        a. IDRiD_Disease Grading_Training Labels.csv
        b. IDRiD_Disease Grading_Testing Labels.csv
```

> **Note:** Datasets are subject to their respective licenses. Use only for research/educational purposes.

---

## Impact and Benefits

### Target Users

| Stakeholder | Benefit |
|---|---|
| **Rural Diabetic Patients** | Earlier detection → Faster treatment → Vision preservation |
| **Primary Health Centers** | AI-assisted screening enables examination of more patients |
| **ASHA Workers** | Simple tool for first-level screening in community settings |
| **Tele-Ophthalmologists** | Explainable predictions enable faster, confident reviews |
| **District Health Programs** | Workflow simulation helps optimize resources and planning |

### Value Proposition

**End-to-End Impact:**
```
Rural Fundus Capture → AI Quality Check → Automated Screening 
→ Visual Explanation → Clinician Validation → Timely Referral 
→ Early Treatment → Vision Preservation
```

**Key Advantages:**
- ✅ Built on trusted MATLAB platform (familiar to engineers & researchers)
- ✅ Explainable AI increases clinician trust and adoption
- ✅ Quality checks prevent analysis of unusable images
- ✅ Designed for resource-constrained rural settings
- ✅ Simulink workflow modeling enables deployment planning

---

## Future Roadmap

### Phase 1: Validation & Refinement
- [ ] External dataset validation (APTOS, Messidor-2)
- [ ] Quantitative Grad-CAM evaluation against lesion masks
- [ ] Clinical validation with ophthalmologist feedback

### Phase 2: Enhanced Features
- [ ] Real-time lesion detection and localization
- [ ] Multi-disease screening (glaucoma, AMD)
- [ ] Mobile app integration for field workers

### Phase 3: Deployment
- [ ] Simulink telemedicine workflow implementation
- [ ] Cloud deployment for PHC connectivity
- [ ] Field pilot in rural health camps
- [ ] Integration with existing health information systems

### Phase 4: Scale & Impact
- [ ] Multi-language support (Hindi, regional languages)
- [ ] Training programs for ASHA workers
- [ ] Government health department partnerships
- [ ] National-scale deployment planning

---

## Team

**Team Name:** RetiNova

**Team Members:**
1. **Harsh Kulkarni** — Team Leader
2. Prathamesh Devkar
3. Vihaan Aptekar
4. Sanika Chowdhary
5. Shravani Dhadge
6. Shreyas Gade

**Institution:** [Your Institution Name]

**Contact:** [Team Contact Email]

---

## Licensing and Attribution

### Model Weights
The EfficientNet-B0 backbone is derived from `adarshcod30/drdetect-dr-screening` (Hugging Face), licensed for **research use only**. Not licensed for clinical or commercial deployment.

### Datasets
- **IDRiD**: IEEE DataPort terms of use
- **APTOS 2019**: Kaggle competition rules
- **DRIVE**: Challenge terms and conditions
- **Messidor-2**: ADCIS third-party usage terms

All datasets must be cited appropriately in publications and presentations.

### Code
This project is developed for Smart India Hackathon 2026. Please contact the team for licensing inquiries.

---

## Disclaimer

⚠️ **IMPORTANT MEDICAL DISCLAIMER**

This is a **research prototype** developed for an educational hackathon. It has **NOT** been:
- Clinically validated in real-world settings
- Approved by regulatory authorities (FDA, CDSCO, etc.)
- Certified as a medical device

**DO NOT USE** for:
- Clinical diagnosis or treatment decisions
- Patient triage without clinician oversight
- Replacement of professional ophthalmological examination

All predictions **MUST** be reviewed and confirmed by a qualified ophthalmologist. This system is intended as a **decision support tool** only, not a replacement for medical judgment.

---

**Built with ❤️ for Smart India Hackathon 2026**

*Making quality eye care accessible to rural India through explainable AI*
