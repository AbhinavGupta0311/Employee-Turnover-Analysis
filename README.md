# 👔 Employee Turnover Analysis using Machine Learning (scikit-learn)

[![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML_Pipeline-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![imbalanced-learn](https://img.shields.io/badge/imbalanced--learn-SMOTE-4B8BBE)](https://imbalanced-learn.org/)
[![KMeans](https://img.shields.io/badge/Clustering-KMeans-00897B)](https://scikit-learn.org/stable/modules/clustering.html#k-means)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)](https://jupyter.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Completed-brightgreen)]()

---

## 📋 Table of Contents

1. [Project Overview](#-project-overview)
2. [Business Problem](#-business-problem)
3. [Dataset Information](#-dataset-information)
4. [Tech Stack](#️-tech-stack)
5. [Project Architecture](#-project-architecture)
6. [Workflow](#-project-workflow)
7. [EDA & Clustering](#-eda--clustering)
8. [Model Development](#-model-development)
9. [Model Performance & Results](#-model-performance--results)
10. [Business Interpretation](#-business-interpretation)
11. [Retention Strategy](#-retention-strategy)
12. [Visualizations](#-visualizations)
13. [Installation & Setup](#️-installation--setup)
14. [Usage](#-usage)
15. [Repository Structure](#-repository-structure)
16. [Future Improvements](#-future-improvements)
17. [Author](#-author)

---

## 📌 Project Overview

Employee turnover is one of the most costly challenges for organizations. This project builds an end-to-end Machine Learning pipeline to **predict which employees are at risk of leaving**, enabling HR teams to take targeted, data-driven retention actions.

Key challenges addressed:
- 📊 **Class Imbalance** — ~23.8% attrition rate handled using SMOTE inside a pipeline
- 🔍 **Unsupervised Clustering** — KMeans used to identify distinct exit behavior patterns
- 🏆 **Model Comparison** — Logistic Regression vs. Random Forest vs. Gradient Boosting
- 🎯 **Probability-Based Risk Zones** — Employees segmented into Green / Yellow / Orange / Red risk tiers

---

## 💼 Business Problem

Organizations lose significant revenue and productivity when employees resign unexpectedly. HR teams need a system that:

- Identifies employees most likely to leave **before** they resign
- Understands **why** employees leave (satisfaction, workload, tenure)
- Prioritizes **who** to focus retention efforts on

This project delivers that by training classifiers on historical HR data and generating actionable retention recommendations.

---

## 📂 Dataset Information

| Property        | Details                              |
|-----------------|--------------------------------------|
| **File**        | `1688640705_hr_comma_sep.xlsx`       |
| **Source**      | HR Analytics Dataset                 |
| **Rows**        | 14,999                               |
| **Columns**     | 10                                   |
| **Target**      | `left` (Binary: 0 = Stayed, 1 = Left)|
| **Class 0**     | Stayed (~76.2%)                      |
| **Class 1**     | Left (~23.8%)                        |

### 🔑 Feature Descriptions

| Feature                 | Type        | Description                                      |
|-------------------------|-------------|--------------------------------------------------|
| `satisfaction_level`    | Float       | Employee satisfaction score (0–1)                |
| `last_evaluation`       | Float       | Last performance evaluation score (0–1)          |
| `number_project`        | Integer     | Number of projects assigned                      |
| `average_montly_hours`  | Integer     | Average monthly working hours                    |
| `time_spend_company`    | Integer     | Years spent at the company                       |
| `Work_accident`         | Binary      | Whether employee had a workplace accident        |
| `promotion_last_5years` | Binary      | Promoted in last 5 years                         |
| `sales`                 | Categorical | Department (sales, technical, support, etc.)     |
| `salary`                | Categorical | Salary level (low, medium, high)                 |

> ⚠️ **Class Imbalance Note:** With ~23.8% turnover rate, SMOTE was applied within the training pipeline to prevent data leakage.

---

## 🛠️ Tech Stack

| Category           | Library / Tool             | Purpose                                      |
|--------------------|----------------------------|----------------------------------------------|
| **Language**       | Python 3.x                 | Core programming language                    |
| **Data**           | Pandas, NumPy              | Data loading and manipulation                |
| **Visualization**  | Matplotlib, Seaborn        | EDA plots, feature importance, confusion matrix |
| **ML Models**      | scikit-learn               | Logistic Regression, Random Forest, Gradient Boosting |
| **Imbalance**      | imbalanced-learn (SMOTE)   | Handling class imbalance in training         |
| **Clustering**     | KMeans (scikit-learn)      | Segmenting employees who left                |
| **Validation**     | Silhouette Score           | Evaluating cluster quality                   |
| **Tuning**         | GridSearchCV               | Hyperparameter optimization                  |
| **Environment**    | Jupyter Notebook           | Interactive development                      |

---

## 🏗️ Project Architecture

```
Raw Data (1688640705_hr_comma_sep.xlsx)
        │
        ▼
┌─────────────────────┐
│  Step 1             │
│  Basic Data         │
│  Understanding      │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Step 2             │
│  EDA: Key Drivers   │
│  of Attrition       │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Step 3             │
│  KMeans Clustering  │
│  (Leavers Only)     │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Step 4             │
│  Data Preparation   │
│  + SMOTE Encoding   │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Step 5             │
│  Model Comparison   │
│  (5-Fold CV)        │
│  + Hyperparameter   │
│  Tuning             │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Step 6             │
│  Best Model         │
│  Selection          │
│  ROC + Confusion    │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Step 7             │
│  Retention Strategy │
│  Risk Zones         │
└─────────────────────┘
```

---

## 🔄 Project Workflow

| Step | Task                          | Details                                                                   |
|------|-------------------------------|---------------------------------------------------------------------------|
| 1    | Data Understanding            | Shape (14999 × 10), dtypes, class distribution, zero null values          |
| 2    | EDA                           | Correlation heatmap, distribution plots, project count vs. turnover       |
| 3    | KMeans Clustering             | Cluster leavers by satisfaction & evaluation; Elbow + Silhouette (0.796)  |
| 4    | Data Preparation              | One-hot encoding, stratified 80/20 split, SMOTE inside pipeline           |
| 5    | Model Training + Tuning       | 5-Fold CV for LR, RF, GB; GridSearchCV on RF (Best ROC-AUC: 0.9918)      |
| 6    | Model Evaluation              | ROC Curves, Confusion Matrix, Classification Reports                      |
| 7    | Retention Strategy            | Probability-based risk zones for actionable HR recommendations            |

---

## 🔍 EDA & Clustering

### Key EDA Findings

| Insight | Finding |
|---|---|
| **Strongest predictor** | `satisfaction_level` (correlation: -0.388 with `left`) |
| **Work_accident** | Negative correlation (-0.154) — accident survivors tend to stay |
| **Time at company** | Positive correlation (0.144) — longer tenure slightly increases turnover |
| **Promotions** | Negative correlation (-0.061) — promoted employees tend to stay |

### KMeans Clustering (3 Clusters of Leavers)

| Cluster | Satisfaction | Evaluation | Profile                          |
|---------|-------------|------------|----------------------------------|
| 🔴 0   | Low (~0.41)  | Low (~0.52)| Disengaged, dissatisfied         |
| 🔵 1   | High (~0.81) | High (~0.91)| High performers who left for better opportunities |
| 🟢 2   | Very Low (~0.11) | High (~0.87) | Overworked, burned-out        |

> **Silhouette Score: 0.796** — indicating well-separated, meaningful clusters.

---

## 🧠 Model Development

### Models Evaluated (5-Fold Stratified CV with SMOTE Pipeline)

| Model               | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---------------------|----------|-----------|--------|----------|---------|
| Logistic Regression | 76%      | 0.50      | 0.74   | 0.60     | 0.82    |
| Random Forest       | 99%      | 0.99      | 0.97   | 0.98     | 0.99    |
| Gradient Boosting   | 97%      | 0.92      | 0.94   | 0.93     | 0.99    |

### Random Forest — GridSearchCV Tuning

| Hyperparameter      | Best Value |
|---------------------|------------|
| `n_estimators`      | 200        |
| `max_depth`         | 15         |
| `min_samples_split` | 5          |
| `min_samples_leaf`  | 1          |
| **Best ROC-AUC**    | **0.9918** |

### Top Feature Importances (Tuned Random Forest)

| Rank | Feature                | Importance % |
|------|------------------------|---------------|
| 1    | `satisfaction_level`   | 28.3%         |
| 2    | `time_spend_company`   | 23.6%         |
| 3    | `number_project`       | 15.1%         |
| 4    | `average_montly_hours` | 13.3%         |
| 5    | `last_evaluation`      | 12.9%         |

---

## 📊 Model Performance & Results

### Final Test Set Metrics (Tuned Random Forest)

| Metric       | Score  |
|--------------|--------|
| **Accuracy** | 98.43% |
| **Precision**| 97.85% |
| **Recall**   | 95.52% |
| **F1 Score** | 96.67% |
| **ROC-AUC**  | 0.9947 |

### Classification Report (Test Set)

|               | Precision | Recall | F1-Score | Support |
|---------------|-----------|--------|----------|---------|
| **0 — Stayed**  | 0.99    | 1.00   | 0.99     | 2,286   |
| **1 — Left**    | 0.98    | 0.96   | 0.97     |   714   |

---

## 💼 Business Interpretation

### Why Recall is the Primary Metric

In employee turnover prediction, **False Negatives are the costliest error**:

| Error Type       | Description                                    | Business Impact    |
|------------------|------------------------------------------------|--------------------|
| **False Positive** | Predicted to leave, actually stays           | Wasted retention resources (Medium) |
| **False Negative** | Predicted to stay, actually leaves           | Employee lost with no intervention (**HIGH**) |

> ✅ Maximizing **Recall (Sensitivity)** ensures most at-risk employees are flagged for retention before they resign.

---

## 🎯 Retention Strategy

### Probability-Based Risk Zones (from Best Model)

| Risk Zone            | Turnover Probability | Headcount | Action                                          |
|----------------------|----------------------|-----------|-------------------------------------------------|
| 🟢 Safe (Green)     | < 20%                | 2,245     | Regular engagement & recognition                |
| 🟡 Low Risk (Yellow) | 20% – 60%            | 47        | Targeted training & periodic check-ins          |
| 🟠 Medium (Orange)  | 60% – 90%            | 25        | Personalized retention plans & incentives       |
| 🔴 High Risk (Red)  | > 90%                | 683       | Immediate intervention & compensation review    |

---

## 🖼️ Visualizations

The notebook contains the following visualizations, generated at each step of the analysis:

### 📊 EDA Visualizations (Step 2)

| Plot | Description |
|------|-------------|
| **Correlation Heatmap** | Feature correlation with the `left` target — `satisfaction_level` has the strongest negative correlation (-0.388), `Work_accident` also negatively correlated (-0.154) |
| **Distribution Plots (3-panel)** | Histograms with KDE for `satisfaction_level` (blue), `last_evaluation` (green), and `average_montly_hours` (orange) — reveals bimodal satisfaction distribution |
| **Project Count Bar Chart** | Count of employees by `number_project` grouped by turnover status — employees with only 2 projects show the highest attrition |

> **Key Insight:** Satisfaction shows a bimodal pattern — a large group of very dissatisfied employees and a normal group of satisfied ones. This strongly separates leavers from stayers.

---

### 🔵 Clustering Visualizations (Step 3)

| Plot | Description |
|------|-------------|
| **Elbow Method Chart** | WCSS plotted for K=1 to 10 — optimal cluster count identified at **K=3** |
| **KMeans Scatter Plot** | Employees who left, plotted by `satisfaction_level` vs `last_evaluation`, coloured by cluster (3 distinct exit patterns) |
| **Cluster Bar Chart** | Average values of satisfaction, evaluation, projects, hours, and tenure per cluster — clear behavioral separation |

> **Cluster Centers:**
> - Cluster 0: Satisfaction=0.41, Evaluation=0.52 → *Disengaged, low performers*
> - Cluster 1: Satisfaction=0.81, Evaluation=0.91 → *High performers leaving for better opportunities*
> - Cluster 2: Satisfaction=0.11, Evaluation=0.87 → *Burned-out overworked employees*
>
> **Silhouette Score: 0.796** — confirms well-separated clusters.

---

### 🏆 Feature Importance (Step 5)

| Plot | Description |
|------|-------------|
| **Top 10 Feature Importance Bar Chart** | Horizontal bar chart of the tuned Random Forest's feature importances — `satisfaction_level` (28.3%) is the dominant predictor, followed by `time_spend_company` (23.6%) |

---

### 📈 Model Evaluation Plots (Step 6)

| Plot | Description |
|------|-------------|
| **ROC Curve — Logistic Regression** | AUC = 0.82 — baseline performance |
| **Confusion Matrix — Logistic Regression** | Annotated heatmap showing FP/FN distribution |
| **ROC Curve — Random Forest** | AUC = 0.99 — near-perfect discrimination |
| **Confusion Matrix — Random Forest** | Near-perfect classification, minimal false negatives |
| **ROC Curve — Gradient Boosting** | AUC = 0.99 — comparable to Random Forest |
| **Confusion Matrix — Gradient Boosting** | Strong performance across both classes |
| **Final Confusion Matrix (Tuned RF)** | Test set evaluation — 98.43% accuracy |
| **Classification Report Heatmap** | Precision / Recall / F1-Score per class visualized as a colour-coded matrix |

---

### 🎯 Risk Zone Visualization (Step 7)

| Plot | Description |
|------|-------------|
| **Risk Zone Distribution** | Employee counts by risk tier: 🟢 Safe (2,245) · 🟡 Low (47) · 🟠 Medium (25) · 🔴 High (683) |

> **Actionable Output:** The model assigns each employee a turnover probability, mapped to a risk tier to guide targeted HR intervention.

---

## ⚙️ Installation & Setup

### Prerequisites
- Python 3.8+
- pip

### Step 1 — Clone the Repository
```bash
git clone https://github.com/AbhinavGupta0311/Employee-Turnover-Analysis.git
cd Employee-Turnover-Analysis
```

### Step 2 — Create a Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3 — Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4 — Launch Jupyter Notebook
```bash
jupyter notebook
```

Open `Employee_Turnover_Analytics.ipynb` and run all cells.

---

## 🚀 Usage

### Running the Full Pipeline
```python
# Step 1–7 are self-contained in the notebook.
# Run cells top-to-bottom for full reproducibility.
```

---

## 📁 Repository Structure

```
Employee-Turnover-Analysis/
│
├── Employee_Turnover_Analytics.ipynb   # Main analysis notebook
├── 1688640705_hr_comma_sep.xlsx        # HR dataset
├── requirements.txt                    # Python dependencies
├── README.md                           # Project documentation
└── .gitignore                          # Ignored files
```

---

## 🔮 Future Improvements

### High Priority
- [ ] Deploy model as a REST API for HR dashboard integration
- [ ] Add SHAP values for individual-level explainability

### Medium Priority
- [ ] Experiment with XGBoost / LightGBM
- [ ] Build interactive retention dashboard (Streamlit / Dash)

### Nice to Have
- [ ] Real-time prediction pipeline
- [ ] Time-series analysis of employee satisfaction trends

---

## 👤 Author

**Abhinav Gupta**
- GitHub: [@AbhinavGupta0311](https://github.com/AbhinavGupta0311)

---

## 🙏 Acknowledgements

- scikit-learn documentation and community
- imbalanced-learn library for SMOTE implementation
- Kaggle HR Analytics community for dataset inspiration
