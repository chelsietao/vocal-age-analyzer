# AI Vocal Age Analyzer
### Vocal Biomarker Detection with Explainable Machine Learning

[![Python](https://img.shields.io/badge/Python-3.10-blue)]()
[![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange)]()
[![SHAP](https://img.shields.io/badge/XAI-SHAP-green)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)]()

A machine learning system that predicts speaker age group from acoustic 
features, grounded in clinical Presbyphonia literature and validated through 
SHAP-based explainability analysis.

**Course Project** | Introduction to AI Applications and Technology  
**Institution** | National Tsing Hua University

---

## Motivation

Standard health AI systems return predictions without explanation — 
a critical failure in medical contexts. This project addresses two questions:

1. Can vocal acoustic features serve as non-invasive digital biomarkers 
   of biological aging?
2. Can the model's decision logic be validated against established 
   medical theory using XAI methods?

---

## System Pipeline

    Raw Audio (.wav)
          ↓
    Feature Extraction (praat-parselmouth + librosa)
          ↓
    17-Dimensional Acoustic Feature Vector
    [F0, Jitter, Shimmer, HNR, MFCC_1 ~ MFCC_13]
          ↓
    XGBoost Classifier (3-class: Young / Middle / Senior)
          ↓
    SHAP Explainability Audit
          ↓
    Gradio Web Application (live microphone input)
---


## Medical Foundation

Each feature corresponds to a documented mechanism of vocal fold aging 
(Presbyphonia):

| Feature | Aging Mechanism | Direction |
|---------|----------------|-----------|
| F0 mean | Vocal fold atrophy / edema | ↑ males, ↓ females |
| Jitter | Neuromotor control decline | ↑ |
| Shimmer | Reduced muscle strength | ↑ |
| HNR | Incomplete glottal closure | ↓ |
| MFCC 1–13 | Vocal tract resonance changes | varies |

---

## Dataset

**CREMA-D** (Crowd-sourced Emotional Multimodal Actors Dataset)
- 91 actors, ages 20–74
- 7,442 audio files (.wav)
- Source: [Kaggle](https://www.kaggle.com/datasets/ejlok1/cremad)

| Age Group | Range | Count | Proportion |
|-----------|-------|-------|------------|
| Young | 20–35 | 4,168 | 56% |
| Middle | 36–50 | 1,880 | 25% |
| Senior | 51+ | 1,394 | 19% |

---

## Results

| Model | Accuracy | Senior F1 | Young F1 |
|-------|----------|-----------|----------|
| Random Forest | 61% | 0.46 | 0.72 |
| **XGBoost** | **66%** | **0.49** | **0.77** |
| Random Baseline | 38% | — | — |

Both models exhibit **ordinal confusion patterns** — misclassifications 
concentrate between adjacent age groups, confirming the model learned 
the continuous nature of vocal aging rather than arbitrary class boundaries.

---

## SHAP Validation

![SHAP Summary](results/shap_summary.png)

![SHAP Beeswarm](results/shap_beeswarm.png)

Key finding: **Low HNR systematically pushes predictions toward Senior**, 
directly corroborating the medical theory of incomplete glottal closure 
in aged vocal folds. The model's decision logic is scientifically auditable.

F0 shows mixed directional influence, consistent with the opposing aging 
trajectories in males (↑) and females (↓) within a mixed-gender dataset.

---

## Confusion Matrix

![Confusion Matrix](results/confusion_matrix.png)

---

## Project Structure

    vocal-age-analyzer/
    ├── notebooks/
    │   ├── voice_age_project_0428.ipynb  # Training pipeline
    │   └── voice_age_demo.ipynb          # Gradio demo
    ├── results/                          # Output figures
    ├── docs/                             # Full report (PDF)
    ├── requirements.txt
    ├── .gitignore
    └── LICENSE

> **Note:** Audio files (AudioWAV/), extracted features (features.csv),
> and trained model (model_artifacts.pkl) are not included due to file
> size constraints. Run the training notebook to reproduce all artifacts.

---

## Reproducing the Results

```bash
# 1. Clone the repository
git clone https://github.com/chelsietao/vocal-age-analyzer.git
cd vocal-age-analyzer

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download CREMA-D dataset
kaggle datasets download -d ejlok1/cremad

# 4. Run the training notebook
# Open notebooks/voice_age_project_0428.ipynb in Google Colab
# Follow the cell execution order documented in the notebook

# 5. Launch the demo
# Open notebooks/voice_age_demo.ipynb in Google Colab
# Run all cells to generate a public Gradio link
```

---

## Future Directions

- **Clinical Extension:** Swap CREMA-D for Parkinson's speech datasets
  to reframe as early neurodegeneration screening
- **Architecture Upgrade:** ResNet18 on Mel-spectrograms with Grad-CAM
  for spatial frequency attention visualization
- **Gender-stratified Modeling:** Separate models for male and female
  speakers to recover F0's directional aging signal

---

## Version History

| Version | Description |
|---------|-------------|
| v1 | Decision tree with mock labels. Accuracy 15% (below random). Pipeline validation only. |
| v2 | Real CREMA-D labels, 17-dim features, XGBoost + SHAP. Accuracy 66%. |

---

## References

1. Cao et al. (2014). CREMA-D. *IEEE Trans. Affective Computing*, 5(4).
2. Kendall & Singh (2021). Presbyphonia. *StatPearls*.
3. Lundberg & Lee (2017). SHAP. *NeurIPS 30*.
4. Chen & Guestrin (2016). XGBoost. *KDD 2016*.
5. Boersma & Weenink (2023). Praat v6.3.
6. Davis & Mermelstein (1980). MFCC. *IEEE TASLP*, 28(4).

