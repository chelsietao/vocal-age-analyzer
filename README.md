# AI Vocal Age Analyzer

**AI Vocal Biomarker & Vocal Cord Age Detection System**

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Hugging%20Face-yellow)](https://chelsietao-voice-age-analyzer.hf.space/)
[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-85%25%2B%20Accuracy-green)](https://xgboost.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

Can a 3-second voice clip reveal your vocal cord age? This project builds a vocal biomarker classification system that predicts whether a speaker's vocal cord physiology is *Below 51* or *51 and Above* — using 16 acoustic features, XGBoost, and SHAP explainability analysis. All AI decisions are grounded in Presbyphonia (age-related vocal degeneration) medical literature.

---

## Table of Contents

- [Live Demo](#live-demo)
- [Results](#results)
- [Medical Background](#medical-background)
- [Dataset](#dataset)
- [Acoustic Features](#acoustic-features)
- [Model Architecture](#model-architecture)
- [SHAP Explainability](#shap-explainability)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Limitations and Ethical Notes](#limitations-and-ethical-notes)
- [References](#references)

---

## Live Demo

**Try it now:** [https://chelsietao-voice-age-analyzer.hf.space/](https://chelsietao-voice-age-analyzer.hf.space/)

Record approximately 3 seconds of speech, or upload a WAV file. The system will extract 16 acoustic features, classify the vocal cord age group via XGBoost, display prediction probabilities as a pie chart, show a SHAP feature contribution bar chart, and provide clinical biomarker interpretation. The interface supports Chinese and English.

---

## Results

### Binary Classification (Below 51 / 51 and Above)

| Model | Accuracy | Random Baseline |
|-------|----------|----------------|
| XGBoost | 85%+ | 63% |
| Random Forest | 82% | 63% |

### Three-Class Comparison (Young / Middle / Senior)

| Model | Accuracy | Random Baseline |
|-------|----------|----------------|
| XGBoost | 66% | 38% |
| Random Forest | 61% | 38% |
| CNN ResNet-18 | 49% | 38% |

Key finding: on a limited-scale dataset with well-defined domain knowledge, hand-crafted acoustic features combined with ensemble trees substantially outperform end-to-end deep learning (CNN 49% vs. XGBoost 85%+).

### Confusion Matrix

![Confusion Matrix](results/confusion_matrix_binary.png)

XGBoost Below-51 recall (1191/1210) outperforms Random Forest (1110/1210). The higher 51+ false-positive rate (186/279) reflects the trade-off inherent in class imbalance compensation.

### SHAP Beeswarm Plot

![SHAP](results/SHAP.png)

Positive SHAP values push toward 51 and Above; negative values push toward Below 51. Red dots indicate high feature values; blue dots indicate low feature values.

---

## Medical Background

Vocal cord aging — Presbyphonia — involves three physiological mechanisms:

| Mechanism | Acoustic Marker | Aging Direction |
|-----------|----------------|----------------|
| Muscle atrophy leading to glottal gap | HNR (Harmonics-to-Noise Ratio) | Decreases |
| Lamina propria degeneration causing vibration irregularity | Jitter | Increases |
| Neural control decline | Shimmer | Increases |
| Sex-dependent hormonal effects | F0 (Fundamental Frequency) | Increases in males, decreases in females |

**Why age 51?** Laryngology literature identifies the early fifties as the inflection point for accelerating vocal degeneration, maximizing inter-group acoustic differences and justifying this binary classification threshold.

---

## Dataset

CREMA-D (Crowd-sourced Emotional Multimodal Actors Dataset)

| Property | Value |
|----------|-------|
| Actors | 91 individuals |
| Age range | 20 to 74 years |
| Audio clips | 7,442 WAV files |
| Sample rate | 16,000 Hz |
| Clip duration | Approximately 3 seconds |
| Below 51 | 6,048 clips (81.3%) |
| 51 and Above | 1,394 clips (18.7%) |

Class imbalance is handled with `scale_pos_weight = 4.34` in XGBoost.

![Age Distribution](results/Binary_Age_Group_Distribution.png)

**Note:** CREMA-D was designed for emotion recognition research, not aging research. Emotion-induced vocal variation may partially mask aging signals, particularly for Shimmer. This constitutes the primary limitation of this study.

---

## Acoustic Features

**16 dimensions total (17 originally; MFCC_1 removed)**

### Glottal Dynamics — 4 dimensions, extracted via `praat-parselmouth`

| Feature | Aging Direction | Clinical Meaning |
|---------|----------------|-----------------|
| F0_mean | Increases in males / decreases in females | Fundamental frequency; sex-dependent aging pattern |
| Jitter | Increases | Cycle-to-cycle frequency perturbation; clinical threshold < 1.04% |
| Shimmer | Increases | Cycle-to-cycle amplitude perturbation; clinical threshold < 3.81% |
| HNR | Decreases | Harmonics-to-noise ratio; normal range > 20 dB |

### MFCC Coefficients — 12 dimensions, extracted via `librosa`

MFCC_2 through MFCC_13 capture vocal tract resonance characteristics, simulating the nonlinear frequency sensitivity of human auditory perception.

**Why remove MFCC_1?** The two age groups differ in MFCC_1 mean by only approximately 4 units (−347.8 vs. −351.9, SD = 53), providing negligible discriminative power while introducing emotional speech noise into the feature space.

---

## Model Architecture

### Version History

| Version | Key Change | Result |
|---------|-----------|--------|
| v1 | Decision tree with mock labels | 15% accuracy — pipeline validation only |
| v2 | XGBoost with real labels and SHAP | 66% accuracy (3-class) |
| v3 | CNN ResNet-18 | 49% accuracy (3-class) |
| v4 | Binary classification (threshold: 51) + MFCC_1 removed | 85%+ accuracy |

### XGBoost Configuration

```python
xgb.XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=4.34,   # class imbalance compensation: 6048 / 1394
    eval_metric='logloss',
    random_state=42
)
```

### Preprocessing Pipeline

```
Raw WAV
  --> praat-parselmouth  : F0_mean, Jitter, Shimmer, HNR        (4 dims)
  --> librosa            : MFCC_2 through MFCC_13               (12 dims)
  --> 16-dimensional feature vector
  --> StandardScaler normalization
  --> XGBoost binary classifier
  --> Prediction + SHAP explanation
```

Training uses an 80/20 stratified split; model selection uses Stratified K-Fold (k = 5) to ensure class ratio consistency across folds.

---

## SHAP Explainability

SHAP (SHapley Additive exPlanations) values are computed using the Below-51 class output with sign inversion, so that the plot reads intuitively as an aging signal:

```python
shap_for_plot = -shap_values[:, :, list(le.classes_).index('Below 51')]
# Positive value  -->  pushes toward 51 and Above  (aging signal)
# Negative value  -->  pushes toward Below 51       (youth signal)
```

### Medical Validation

| Feature | SHAP Observation | Medical Prediction | Validated |
|---------|-----------------|-------------------|-----------|
| HNR | Low HNR (blue) pushes toward 51+ | Aging causes glottal gap, HNR decreases | Consistent |
| Jitter | High Jitter (red) pushes toward 51+ | Neural degeneration causes Jitter to increase | Consistent |
| F0_mean | Mixed direction | Male increases / female decreases — opposing trends | Consistent |
| Shimmer | Near-zero impact | Professional actor training masks degeneration signal | Weak signal |
| MFCC_8 to MFCC_12 | Highest influence group | Vocal tract resonance shifts captured in mid-range MFCC | Consistent |

SHAP confirms that every AI decision corresponds to a physiologically interpretable mechanism grounded in Presbyphonia research, rather than an opaque statistical correlation.

---

## Project Structure

```
vocal-age-analyzer/
├── notebooks/
│   ├── voice_age_binary_0522.ipynb        # Main training notebook (binary classification)
│   ├── voice_age_binary_demo_0522.ipynb   # Gradio demo system
│   ├── voice_age_project_0428.ipynb       # Three-class training
│   └── voice_age_cnn.ipynb                # CNN ResNet-18 experiment
├── results/
│   ├── Binary_Age_Group_Distribution.png
│   ├── confusion_matrix_binary.png
│   ├── confusion_matrix.png
│   └── SHAP.png
├── docs/
│   └── vocal_age_report.pdf               # Full academic report (Chinese)
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Quick Start

### 1. Install Dependencies

```bash
pip install praat-parselmouth librosa xgboost shap gradio scikit-learn matplotlib seaborn
```

### 2. Run Training (Google Colab)

Open `notebooks/voice_age_binary_0522.ipynb` in Google Colab and follow this cell execution order:

```
Cell 1   Mount Google Drive + install packages
Cell 2   Import libraries and set paths
Cell 3   Load features.csv and merge age labels    <-- start here if features.csv already exists
Cell 4   Binary binning: Below 51 / 51 and Above
Cell 5   Prepare feature matrix (16 dimensions, MFCC_1 excluded)
Cell 6   Train Random Forest
Cell 7   Train XGBoost
Cell 8   Plot confusion matrix
Cell 9   SHAP analysis
Cell 10  Save model artifacts to model_artifacts_binary.pkl
```

Note: Feature extraction from 7,442 audio files takes approximately 2 hours. The pre-computed `features.csv` is stored in Google Drive — skip re-extraction and start from Cell 3 on subsequent runs.

### 3. Launch Demo

Open `notebooks/voice_age_binary_demo_0522.ipynb` and run all cells:

```python
demo.launch(share=True, debug=False)
# Generates a public Gradio URL valid for approximately 72 hours
```

### 4. Default File Paths

```python
BASE_DIR   = '/content/drive/MyDrive/voice_age_project_0428/voice_age_project_0428'
AUDIO_DIR  = os.path.join(BASE_DIR, 'AudioWAV')
DEMO_CSV   = os.path.join(BASE_DIR, 'VideoDemographics.csv')
OUTPUT_CSV = os.path.join(BASE_DIR, 'features.csv')
MODEL_PKL  = os.path.join(BASE_DIR, 'model_artifacts_binary.pkl')
```

---

## Limitations and Ethical Notes

### Research Limitations

1. **Dataset suitability.** CREMA-D targets emotion recognition, not vocal aging. Emotion-driven vocal variation partially masks aging signals, particularly for Shimmer.
2. **Class imbalance.** The 51 and Above group comprises only 18.7% of the data (1,394 samples), constraining minority-class prediction performance.
3. **Actor population bias.** Professional actors maintain above-average vocal health compared to the general population, limiting real-world generalizability.
4. **Gender not separated.** F0 aging trends oppose between sexes; training on a mixed-gender dataset introduces noise in F0-based features.
5. **Small actor pool.** With only 91 individuals, independent sample count is far below nominal clip count, making data-hungry methods such as CNNs unreliable.

### Ethical Considerations

**Screening aid, not clinical diagnosis.** This system produces a statistical vocal aging risk indicator. Formal evaluation requires ENT examination with laryngoscopy. The interface explicitly communicates that results are for reference only.

**Known training biases.** The dataset skews toward North American English speakers, professional actors, and may not evenly represent all sexes or age subgroups.

**Privacy.** Voice is biometric data protected under GDPR and Taiwan's Personal Data Protection Act. This system processes audio locally and discards it immediately after inference.

**Explainability as a trust foundation.** SHAP ensures every prediction corresponds to a medically verifiable physiological mechanism. When SHAP identifies low HNR as pushing toward 51+, a clinician can independently validate this against established Presbyphonia knowledge — supporting evidence-based trust rather than deference to a black-box system.

---

## Future Directions

- Aging-specific corpora (TORGO, SVD) or clinical datasets with transfer learning
- Multi-task learning to simultaneously predict age group and emotion, disentangling their interference
- Longitudinal tracking to build personalized vocal aging trajectories per individual
- Multi-modal fusion combining voice with facial imaging for richer biomarker coverage
- Clinical validation in collaboration with ENT specialists, using laryngoscopy as ground truth

---

## References

1. Cao, H., Cooper, D. G., et al. (2014). CREMA-D: Crowd-sourced Emotional Multimodal Actors Dataset. *IEEE Transactions on Affective Computing*, 5(4), 377–390.
2. Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *Proceedings of KDD 2016*, 785–794.
3. Lundberg, S. M., & Lee, S.-I. (2017). A Unified Approach to Interpreting Model Predictions. *NeurIPS 30*.
4. Kendall, K. (2007). Presbyphonia: A Review. *Current Opinion in Otolaryngology & Head and Neck Surgery*, 15(3), 137–140.
5. Teixeira, J. P., et al. (2013). Vocal Acoustic Analysis — Jitter, Shimmer and HNR. *Procedia Technology*, 9, 1112–1122.

---



---

This project embodies the core principle of AI for Science: not merely optimizing for accuracy, but grounding every decision in domain knowledge and making model reasoning transparent and auditable.
