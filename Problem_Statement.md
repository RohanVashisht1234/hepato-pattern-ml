# 📋 Machine Learning Case Study — Problem Statement

<p align="center">
  <b>B.Tech CSE (2024–2028) • Semester V • Course: Machine Learning</b><br>
  <b>Case Study No. 53 — Disease Pattern Classification</b>
</p>

---

## 🎯 Case Study Prompt

> **"A healthcare dataset contains observations associated with different health conditions. Develop a system to investigate and classify the relevant outcomes."**  
> *(With Proper Justification)*

### 🏷️ Student-Formulated Project Title:
**HepatoPattern: Classification of Liver Cirrhosis & Disease Progression from Biochemical Blood Biomarkers**

---

## 📌 Project Objectives

* [x] **Problem Formulation:** Understand and formally define the assigned real-world machine learning problem with strong clinical justification.
* [x] **Dataset Identification & Justification:** Identify, audit, and justify an authentic healthcare benchmark (*Hannover Medical School Hepatitis C & Cirrhosis Cohort*).
* [x] **Formulation & Task Type:** Formulate the required supervised clinical risk classification task (Active Liver Pathology vs. Healthy Controls).
* [x] **Data Quality & Preprocessing:** Perform missing value audit, median imputation, categorical encoding, and zero-leakage feature standardization.
* [x] **Model Development & Benchmarking:** Implement and compare 3 distinct ML paradigms (*Logistic Regression*, *Decision Trees*, *Random Forest*).
* [x] **Rigorous Evaluation:** Benchmark models using Repeated Stratified 5-Fold Cross-Validation, ROC-AUC, Recall, Precision, and Confusion Matrix analysis.
* [x] **Clinical Feature Experiment:** Compare Full Clinical Blood Panel (12 features) vs. Routine Outpatient LFT Panel (6 features).
* [x] **Interactive Streamlit Application:** Build and deploy a dual-mode interactive clinical decision support web application (*Bedside Nurse Intake Form + Research Dashboard*).

---

## 🏆 Expected Outcomes

| Outcome Area | Description | Status |
|---|---|:---:|
| **1. Defined Problem** | A documented machine learning problem with clear clinical cost rationale and target definition. | ✅ Achieved |
| **2. Prepared Dataset** | Documented dataset source (UCI #571), 12 biochemical biomarkers, and leakage-free preprocessing pipeline. | ✅ Achieved |
| **3. Exploratory Analysis** | Comprehensive EDA showing demographic distributions, enzyme boxplots, and correlation heatmaps. | ✅ Achieved |
| **4. Evaluated ML Models** | Benchmark comparison of 3 models with tuned Random Forest champion (97.6% Test Accuracy, 0.997 ROC-AUC). | ✅ Achieved |
| **5. Error & Feature Analysis** | Confusion matrix inspection, false negative mitigation via threshold tuning, and Gini feature importances. | ✅ Achieved |
| **6. Working Prototype** | Production-ready Streamlit application with intuitive bedside triage form and live visualizations. | ✅ Achieved |

---

## 📂 Deliverables & Project Mapping

| Deliverable Required | What Was Accomplished | Location in Project |
|:---|:---|:---|
| **Problem Definition** | Formal clinical problem statement, triage cost rationale (minimizing False Negatives), and student title. | `README.md`, Notebook Section 1 |
| **Dataset & Documentation** | UCI Hannover Medical School benchmark (615 patient records, 12 biochemical biomarkers). | `cirrhosis.csv`, Notebook Section 2 |
| **Exploratory Data Analysis (EDA)** | Distribution histograms, stage distributions, and cross-feature biomarker correlations. | Notebook Section 3 |
| **Preprocessing Pipeline** | Leakage-free `SimpleImputer(median)` $\rightarrow$ `StandardScaler()` $\rightarrow$ `Classifier` pipeline. | Notebook Section 4 |
| **Model Development** | Linear (Logistic Regression), Tree-based (Decision Tree), and Advanced Ensemble (Random Forest). | Notebook Section 5 |
| **Feature Panel Experiment** | Full Clinical Panel (12 features) vs. Routine Outpatient Panel (6 features). | Notebook Section 6 |
| **Evaluation & Analysis** | Test accuracy (97.6%), ROC-AUC (0.997), confusion matrix, and threshold optimization ($0.50 \rightarrow 0.30$). | Notebook Section 7 |
| **Feature Importance** | Biomarker ranking (AST, ALT, GGT, CHE, BIL) with clinical physiological rationale. | Notebook Section 8 |
| **Interactive Streamlit App** | Dual-interface web application (Nurse Bedside Triage Form + Clinical Research Dashboard). | `streamlit_deployment/main.py` |
| **Final Documentation** | Comprehensive project report and interactive Google Colab notebook. | `Documentation.pdf`, `Notebook.ipynb` |

---

<p align="center">
  <b>Case Study No. 53 • Machine Learning Examination & Submission</b><br>
  <i>HepatoPattern — End-to-End Clinical Intelligence System</i>
</p>