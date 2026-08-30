# RetiNova — Explainable AI for Diabetic Retinopathy Screening in Rural India

**Smart India Hackathon 2026**

| | |
|---|---|
| **Problem Statement ID** | 26038 |
| **Theme** | MedTech / BioTech / HealthTech |
| **Category** | Software |
| **Organization** | MathWorks |
| **Team Name** | RetiNova |

---

## Table of Contents

- [Problem Background](#problem-background)
- [What This Project Does](#what-this-project-does)
- [Why This Matters](#why-this-matters)
- [System Pipeline](#system-pipeline)
- [Technical Approach](#technical-approach)
- [Tech Stack](#tech-stack)
- [Design Targets](#design-targets)
- [Feasibility and Challenges](#feasibility-and-challenges)
- [Impact and Benefits](#impact-and-benefits)
- [Datasets](#datasets)
- [Technical Foundations / References](#technical-foundations--references)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Roadmap](#roadmap)
- [Team](#team)

---

## Problem Background

India has more than 77 million diabetic adults, the second highest number in the world. Around 18% of them develop Diabetic Retinopathy (DR), which is one of the leading causes of preventable blindness. Early screening can prevent up to 90% of vision loss caused by DR, but India has only about 1 ophthalmologist per 100,000 people in rural areas. This makes manual, large-scale screening practically impossible.

Most existing AI-based DR screening tools have three big problems:

1. They work as **black boxes** — doctors cannot see *why* the AI made a prediction.
2. They are **not clinically validated** rigorously enough for real deployment.
3. They **break down on real-world images** — fundus photos taken with portable cameras in rural camps are often poorly lit, out of focus, or partially obstructed.

RetiNova is built to solve exactly these three problems.

---

## What This Project Does

RetiNova is a MATLAB and Simulink based pipeline that screens retinal (fundus) images for Diabetic Retinopathy and explains its reasoning to the doctor, instead of just giving a number.

In short, it:

- Checks whether a retinal image is good enough to analyze, and asks for a recapture if not.
- Enhances and cleans the image.
- Detects retinal structures and DR-related lesions (vessels, microaneurysms, exudates, hemorrhages, neovascularization).
- Grades DR severity on the standard 5-level clinical scale (0–4).
- Shows *visual evidence* (via Grad-CAM) for every prediction, so a doctor can verify it in under 30 seconds.
- Simulates the full telemedicine workflow in Simulink to help health departments plan bandwidth, staffing, and doctor review capacity.

---

## Why This Matters

| Problem | How RetiNova Helps |
|---|---|
| Very few ophthalmologists in rural India | AI does the first-level screening, doctors only review flagged/high-risk cases |
| Portable camera images are often low quality | Built-in quality check + enhancement step before analysis |
| Black-box AI reduces doctor trust | Grad-CAM + lesion evidence + confidence score with every result |
| Delayed diagnosis leads to vision loss | Faster triage means faster referral and earlier treatment |
| District health programs lack planning data | Simulink model estimates bandwidth, processing time, and doctor workload |

---

## System Pipeline

The core idea is a straight-line pipeline, from image capture to a doctor's decision:

```
Fundus Image
      │
      ▼
Quality Check  ──── (fails) ──► Recapture Request
      │ (passes)
      ▼
Image Enhancement (CLAHE, denoising, illumination correction)
      │
      ▼
Retinal Analysis (vessels, optic disc/fovea, microaneurysms,
                   exudates, hemorrhages, neovascularization)
      │
      ▼
DR Grading (Level 0 – 4, International Clinical DR Scale)
      │
      ▼
Explainable AI (Grad-CAM heatmap + lesion evidence + confidence score)
      │
      ▼
Clinician Review (< 30 seconds)
      │
      ▼
Referral Decision
```

### 7-Step Technical Methodology

| Step | Stage | What Happens |
|---|---|---|
| 1 | **Image Input** | Retinal images captured using a portable fundus camera at PHCs (Primary Health Centres) or rural health camps |
| 2 | **Quality Check** | Focus, illumination, and field of view are checked. Poor-quality images are flagged for recapture |
| 3 | **Image Processing** | Brightness/contrast correction, noise reduction, and detail enhancement |
| 4 | **Retinal Analysis** | Detection of vessels, optic disc/fovea, microaneurysms, exudates, hemorrhages, and neovascularization |
| 5 | **AI Grading** | Classification into 5 DR severity levels (0–4) using the International Clinical DR Severity Scale |
| 6 | **Explainable AI** | Grad-CAM heatmaps, lesion-level evidence, confidence scores, and an auto-generated annotated report |
| 7 | **Telemedicine Simulation (Simulink)** | Models image flow, bandwidth usage, processing time, and ophthalmologist review capacity |

---

## Technical Approach

**Core Tools:** MATLAB, MATLAB Toolboxes, and Simulink, used for image processing, AI model development, medical imaging analysis, and system-level simulation.

**Toolboxes used:**
- Image Processing Toolbox
- Computer Vision Toolbox
- Deep Learning Toolbox
- Medical Imaging Toolbox
- Simulink
- Statistics and Machine Learning Toolbox

The pipeline is built as a set of modular stages (quality check → enhancement → analysis → grading → explainability), so each stage can be independently tested, validated, and swapped/improved without breaking the rest of the system. The final telemedicine workflow (bandwidth, review time, doctor workload) is modeled separately in Simulink so district health programs can simulate different deployment scenarios before rolling the system out.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Image Processing | MATLAB Image Processing Toolbox |
| Computer Vision / Segmentation | MATLAB Computer Vision Toolbox |
| Deep Learning / DR Grading | MATLAB Deep Learning Toolbox |
| Medical Imaging Specific Tools | MATLAB Medical Imaging Toolbox |
| Statistical Validation | Statistics and Machine Learning Toolbox |
| Workflow / Telemedicine Simulation | Simulink |
| Explainability | Grad-CAM (Gradient-weighted Class Activation Mapping) |
| Clinical Standard | International Clinical DR Severity Scale |

---

## Design Targets

These are the performance goals the system is designed to meet:

| Metric | Target |
|---|---|
| Sensitivity | > 90% |
| Specificity | > 85% |
| Referable DR Threshold | Level 2 and above |
| Doctor Review Time per Image | < 30 seconds |

---

## Feasibility and Challenges

| Challenge | Our Approach |
|---|---|
| Poor image quality from field cameras | Quality check + image enhancement before analysis |
| Tiny lesions (microaneurysms) are hard to detect | Multi-scale lesion analysis |
| Uneven/imbalanced training data | Balanced training strategy + proper evaluation metrics |
| Black-box AI reduces trust | Grad-CAM + lesion-level evidence |
| Limited doctors and network bandwidth in rural areas | Human-in-the-loop review + Simulink-based resource planning |

**Feasibility highlights:**
- **Software:** MATLAB natively supports AI, image processing, and simulation in one environment.
- **Data:** Public datasets (APTOS, IDRiD, DRIVE, Messidor-2) provide real retinal images for training and validation.
- **Deployment:** Designed specifically for portable fundus cameras, PHCs, and telemedicine setups, not high-end hospital equipment.
- **Validation:** Tested on separate datasets and unseen images to avoid overfitting and to check real-world generalization.

---

## Impact and Benefits

**Target Users:** Rural diabetic patients, PHCs & ASHA workers, tele-ophthalmologists, and district health programs.

| Stakeholder | Impact |
|---|---|
| **Rural Diabetic Patients** | Earlier DR detection → faster referral → earlier treatment → lower risk of vision loss |
| **PHCs & ASHA Workers** | AI-assisted screening, more patients screened, doctors focus only on high-risk cases, supports telemedicine |
| **Tele-Ophthalmologists** | Explainable AI, better understanding of the AI's decision, faster result verification, evidence-based decisions |
| **District Health Programs** | Simulink-based resource planning, better bandwidth usage, better doctor workload planning |

**End-to-end value chain:**

```
Rural Fundus Image → AI Screening → Explainable Evidence → Doctor Validation → Early Referral → Vision Preservation
```

---

## Datasets

The following publicly available datasets are used for training and validation:

1. **APTOS 2019 Blindness Detection**
   https://www.kaggle.com/c/aptos2019-blindness-detection

2. **IDRiD — Indian Diabetic Retinopathy Image Dataset**
   https://ieeedataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid

3. **DRIVE — Digital Retinal Images for Vessel Extraction**
   https://drive.grand-challenge.org/

4. **Messidor-2**
   https://www.adcis.net/en/third-party/messidor2/

> Note: These datasets are used strictly for training/validation purposes and are subject to their respective licenses and terms of use.

---

## Technical Foundations / References

- International Clinical DR Severity Scale
- Grad-CAM for explainable AI
- Retinal vessel segmentation techniques
- Fundus image quality assessment methods
- Medical image enhancement (CLAHE, denoising, illumination correction)
- Deep learning for retinal lesion detection
- MATLAB / Simulink based workflow simulation

---

## Project Structure

Structured folder layout of the repository :

```
RetiNova/
├── data/                      # Dataset references / sample images (not the full datasets)
├── quality_check/             # Image quality assessment scripts
├── preprocessing/             # Image enhancement (CLAHE, denoising, etc.)
├── segmentation/              # Vessel, optic disc, lesion detection
├── grading/                   # DR severity classification model
├── explainability/            # Grad-CAM and report generation
├── simulink_model/            # Telemedicine workflow simulation (.slx files)
├── docs/                      # Slide deck, problem statement, documentation
├── results/                   # Sample outputs, validation reports
└── README.md
```

---

## Getting Started

### Prerequisites

- MATLAB (R2023a or later recommended)
- Image Processing Toolbox
- Computer Vision Toolbox
- Deep Learning Toolbox
- Medical Imaging Toolbox
- Statistics and Machine Learning Toolbox
- Simulink

### Steps

1. Clone this repository:
   ```bash
   git clone https://github.com/harsh-k-117/RetiNova.git
   cd RetiNova
   ```
2. Open the project in MATLAB.
3. Download the required datasets (see [Datasets](#datasets)) and place them in the `data/` folder.
4. Run the quality check and preprocessing scripts on a sample image.
5. Run the DR grading model to get a severity classification.
6. Open `simulink_model/` to explore the telemedicine workflow simulation.

---

## Roadmap

- [ ] Finalize image quality assessment module
- [ ] Train and validate DR grading model against target sensitivity/specificity
- [ ] Integrate Grad-CAM explainability module
- [ ] Build and test Simulink telemedicine workflow model
- [ ] Validate against published benchmarks
- [ ] Field testing with sample rural fundus images

---

## Team

**Team Name:** RetiNova
**Problem Statement:** 26038 — Explainable AI for Diabetic Retinopathy Screening in Rural India
**Organization:** MathWorks

**Team Members:**
1. Harsh Kulkarni
2. Prathamesh Devkar
3. Vihaan Aptekar
4. Sanika Chowdhary
5. Shravani Dhadge
6. Shreyas Gade

---
