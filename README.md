# AIoT-Guardian-Self-Healing-AI-for-IoT-Cybersecurity
AIoT-Guardian: Self-Healing Edge Intelligence Framework for Autonomous Cyber Attack Detection in IoT Networks
# AIoT-Guardian

**Self-Healing Edge Intelligence Framework for AIoT Intrusion Detection**

An end-to-end machine-learning pipeline that fuses three public IoT
intrusion-detection datasets, learns a sparse-attention representation,
detects attacks, scores their severity, localizes the affected nodes,
explains predictions, and autonomously remediates + validates recovery.

---

## Pipeline Overview
config.py # all hyper-parameters, paths, shared heuristics
| # | Stage | Module |
|---|-------|--------|
| 1 | Load & combine BETH / BoT-IoT / Edge-IIoTset | `data_pipeline.py` |
| 2 | Stratified 70/15/15 split (leakage-resistant) | `data_pipeline.py` |
| 3 | KNN imputation + One-Hot + Z-score | `data_pipeline.py` |
| 4 | SMOTE (train split only) | `data_pipeline.py` |
| 5 | mRMR feature selection | `data_pipeline.py` |
| 6 | DPSTS dual-Probsparse attention | `models.py` |
| 7 | REFN detector + D-NRO optimizer | `models.py` |
| 8 | FLSM fuzzy severity scoring | `models.py` |
| 9 | Graph-based attack localization | `models.py` |
| 10 | SHAP-style explanations | `main.py` |
| 11 | DDPG-MARL + Immune + PSO remediation | `models.py` |
| 12 | FDRVM recovery validation | `metrics.py` |
Run main.py
---

## Project Layout
