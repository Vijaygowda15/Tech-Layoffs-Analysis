# 🔍 Tech Industry Layoffs Analysis Using Machine Learning

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?logo=scikit-learn&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Neural%20Network-FF6F00?logo=tensorflow&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas&logoColor=white)
![License](https://img.shields.io/badge/License-Academic-lightgrey)

**M.Sc. Dissertation | Data Science and Analytics | JAIN (Deemed-to-be University)**

· Vijay N · 

*Under the guidance of Dr. Santosh Shivraj Singh Chowhan*

---

</div>

## 📌 Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Project Architecture](#project-architecture)
- [Models Implemented](#models-implemented)
- [Results](#results)
- [Feature Importance](#feature-importance)
- [Key Findings](#key-findings)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [How to Run](#how-to-run)
- [Ethical Considerations](#ethical-considerations)
- [Future Work](#future-work)
- [Team](#team)

---

## 📖 Overview

The global tech industry witnessed an unprecedented wave of layoffs between 2020 and 2024 — from pandemic-era over-hiring to post-boom corrections driven by inflation, interest rate hikes, and shifting investor sentiment. Companies like **Amazon, Meta, Google, and Microsoft** collectively laid off tens of thousands of employees.

This project builds a **machine learning pipeline** to predict the **percentage of workforce layoffs** in tech companies using structured company-level and macroeconomic data. By comparing multiple regression and ensemble models, the study identifies the best-performing approach and the most significant predictors of layoff events.

> 💡 **Goal:** Help employees estimate layoff risk at their company, and help companies determine appropriate workforce reduction percentages based on comparable industry data.

---

## ❓ Problem Statement

Traditional HR analytics and economic forecasting tools provide *retrospective* analysis but fall short of offering *predictive foresight*. This research addresses that gap by:

- Predicting the **percentage of employees laid off** (regression task)
- Identifying **key drivers** of layoffs across industries and company types
- Providing **interpretable, explainable** predictions for stakeholders

---

## 📊 Dataset

### Sources

| Dataset | Description |
|--------|-------------|
| **Tech Layoffs 2020–2024** | Kaggle — 3,500+ layoff events across global tech firms ([link](https://www.kaggle.com/datasets/ulrikeherold/tech-layoffs-2020-2024/data)) |
| **NASDAQ-100 Tech Index (NDXT)** | Pulled via `yfinance` — Aug 2019 to Dec 2024 |

### Key Features

| Feature | Description |
|---------|-------------|
| `Company` | Name of the tech company |
| `Industry` | Sector (Fintech, Edtech, SaaS, Retail, etc.) |
| `Stage` | Funding stage (Series A, Post-IPO, Acquired, etc.) |
| `Company_Size_before_Layoffs` | Headcount before the event |
| `Money_Raised_in_$_mil` | Total funding raised |
| `Percentage` | % of workforce laid off *(target variable)* |
| `Stock_Delta` | % change in NDXT index in 90 days before layoff |
| `Region` | US time zone-based region (Pacific, Central, Eastern…) |
| `Date_Layoffs` | Date of the layoff event |

### Data Snapshot — US Layoffs by Year

| Year | Total Laid Off |
|------|---------------|
| 2020 | 44,217 |
| 2021 | 6,137 |
| 2022 | 92,755 |
| 2023 | 1,26,822 |
| 2024 | 4,050 |

### Top 10 Companies by Total Layoffs (2020–2024)

| Company | Employees Laid Off |
|---------|-------------------|
| Amazon | 27,150 |
| Meta | 21,000 |
| Google | 13,000 |
| Microsoft | 10,000 |
| Salesforce | 10,000 |
| Micron | 7,200 |
| Uber | 6,900 |
| Cisco | 4,100 |
| Peloton | 4,084 |
| Carvana | 4,000 |

---

## 🏗️ Project Architecture

```
Dataset (Kaggle + yfinance)
        │
        ▼
   Preprocessing
   ├── Drop nulls & non-US companies
   ├── Min-Max Scaling
   ├── One-Hot Encoding (Stage, Industry, Region)
   ├── UNIX date conversion
   └── Outlier removal (companies < 10 employees)
        │
        ▼
  Feature Engineering
  ├── Stock Delta (90-day NDXT change)
  ├── Time zone-based region mapping
  └── Month / Year / Quarter extraction
        │
        ▼
   80/20 Train-Test Split
        │
   ┌────┴─────────────────────────────┐
   ▼                                  ▼
Training Data                    Testing Data
   │
   ▼
ML Models
├── Linear Regression (Baseline)
├── Polynomial Regression (Degrees 2, 3, 4)
├── ARD Regression (Bayesian)
├── Neural Network (Grid Search + K-Fold CV)
└── Random Forest Regressor ✅ (Best)
        │
        ▼
Performance Evaluation (MSE)
        │
        ▼
Feature Importance + Results Analysis
```

---

## 🤖 Models Implemented

### 1. Linear Regression (Baseline)
A multilinear model used to establish a baseline. Assumes linear relationships between features and layoff percentage — limited by inability to capture non-linear interactions.

### 2. Polynomial Regression (Degrees 2–4)
Extends linear regression with higher-degree terms to model non-linearity. Higher degrees led to overfitting, with testing MSE rising sharply at degree 4.

### 3. ARD Regression (Automatic Relevance Determination)
A Bayesian regression approach that automatically assigns relevance weights to each feature. Improved over linear models by shrinking irrelevant features but constrained by linear assumptions.

### 4. Neural Network
A 4-hidden-layer network with sigmoid activations. Initial performance was poor due to sigmoid being unsuitable for regression. **Grid Search** and **K-Fold Cross Validation (10-fold)** were used to tune hyperparameters — best config: `ReLU`, 3 nodes/layer, LR=0.19.

### 5. Random Forest Regressor ✅ (Best Model)
Ensemble of decision trees with hyperparameter tuning via Grid Search:
- `n_estimators`: 94
- `max_depth`: 20
- `min_samples_split`: 2

Achieved the best balance of accuracy, generalization, and interpretability.

---

## 📈 Results

### Model Comparison (Mean Squared Error)

| Model | Train MSE | Test MSE |
|-------|-----------|----------|
| Linear Regression | 484.89 | 409.13 |
| Polynomial (Degree 2) | 518.52 | 667.18 |
| Polynomial (Degree 3) | 531.32 | 624.80 |
| Polynomial (Degree 4) | 455.39 | 880.05 |
| ARD Regression | 386.46 | 290.17 |
| Neural Network (Base) | 786.07 | 584.98 |
| Neural Network (Grid Search) | 402.80 | 335.09 |
| **Random Forest (Tuned)** | **108.01** | **258.05** |

> ✅ **Random Forest achieved the lowest training MSE (108.01) and best test MSE (258.05)** — outperforming all other models.

---

## 🥧 Feature Importance

The Random Forest model's feature importance analysis revealed the following hierarchy of predictors:

```
Company Size     ████████████████████████  Most Important
Money Raised     ████████████
Industry         ██████████
Date of Layoffs  ████████
Stock Delta      ██████
Region           ███
Stage            █                         Least Important
```

**Key Insight:** Company size dominates predictions — larger companies face higher layoff risk, possibly due to over-expansion. Funding raised and industry sector also play significant roles. Company funding stage has minimal predictive value, suggesting layoffs are more about operational scale than startup lifecycle.

---

## 🔑 Key Findings

- 📈 **2023 was the peak layoff year** in the dataset — 126,822 employees laid off in the US alone
- 🏢 **Consumer, Retail, and Transportation** sectors were most affected
- 💼 **Post-IPO companies** had the most layoffs (312 events) — under pressure from public market expectations
- 🗺️ **San Francisco Bay Area** led with 137,509 layoffs, followed by Seattle (45,056) and NYC (24,367)
- 📉 **Layoff % is weakly correlated** with company size (r=0.11) and money raised (r=0.07) — meaning layoffs happen across all scales
- 🌐 **Stock market performance** (NDXT index) aligns with key economic events: COVID crash (2020), stimulus rally (2021), rate-hike downturn (2022), recovery (2023–24)

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python 3.8+ |
| Data Processing | Pandas, NumPy |
| ML Models | Scikit-Learn (LinearRegression, PolynomialFeatures, ARDRegression, RandomForestRegressor) |
| Neural Networks | TensorFlow / Keras, Keras Tuner |
| Financial Data | yfinance |
| Visualization | Matplotlib, Seaborn |
| Hyperparameter Tuning | GridSearchCV, Keras Tuner |
| Cross Validation | RepeatedKFold (sklearn) |
| Notebook | Jupyter Notebook |

---

## 📁 Project Structure

```
📦 tech-layoffs-ml/
├── 📓 final_project.ipynb          # Main Jupyter Notebook (full pipeline)
├── 📊 tech_layoffs.csv             # Primary layoff dataset
├── 📈 stock_data.csv               # NDXT index data from yfinance
├── 📉 all_model_predictions.csv    # Predictions from all trained models
├── 🖼️ FeatureImportance.png        # Random Forest feature importance chart
├── 📄 Dissertation.pdf             # Full academic dissertation
├── 📊 layoffs_presentation.pptx    # Project presentation slides
└── 📝 README.md                    # You are here
```

---

## ▶️ How to Run

### Prerequisites

```bash
pip install pandas numpy scikit-learn tensorflow keras-tuner yfinance matplotlib seaborn jupyter
```

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/tech-layoffs-ml.git
cd tech-layoffs-ml

# 2. Launch Jupyter Notebook
jupyter notebook final_project.ipynb
```

### Notebook Sections

| Section | Description |
|---------|-------------|
| `Data Loading` | Load layoffs CSV + fetch NDXT stock data via yfinance |
| `EDA` | Layoffs by year, industry, region, company stage |
| `Preprocessing` | Scaling, encoding, outlier removal, train-test split |
| `Model 1: Regression` | Linear → Polynomial → ARD regression |
| `Model 2: Neural Network` | Base NN → K-Fold CV → Grid Search tuning |
| `Model 3: Random Forest` | RF Regressor → Grid Search → Feature Importance |
| `Results` | MSE comparison across all models |

---

## ⚖️ Ethical Considerations

Using ML to predict job losses carries important responsibilities:

- 🔍 **Interpretability** — Random Forest feature importance and SHAP values help stakeholders understand predictions, not just accept them
- ⚠️ **Probabilistic, not deterministic** — Predictions are risk indicators, not guarantees
- 🔒 **No individual targeting** — Models operate at the company level, not the employee level
- ⚖️ **Fairness** — Predictions should be audited for bias across regions, industries, and company types
- 🧭 **Decision support, not decision replacement** — These tools should inform, not automate, workforce decisions

---

## 🔭 Future Work

- [ ] **Expand to Indian tech ecosystem** — Use BSE/Nifty IT indices, Inc42 data, and India-specific macro indicators
- [ ] **Integrate macroeconomic indicators** — GDP, CPI inflation, repo rates, FDI inflows, VC funding trends
- [ ] **Company-level financial features** — Quarterly earnings, P&L ratios, R&D spending
- [ ] **Temporal models** — LSTM / Transformer architectures to capture sequential trends
- [ ] **NLP integration** — Mine Glassdoor reviews, news sentiment, LinkedIn job postings as leading indicators
- [ ] **SHAP explainability** — Case-by-case prediction explanations for stakeholder trust
- [ ] **Interactive web tool** — Streamlit/Dash app for real-time layoff risk scoring

---


**Guide:** Dr. Santosh Shivraj Singh Chowhan, Assistant Professor & Programme Head – PG  
**Institution:** JAIN (Deemed-to-be University), Department of Data Analytics and Mathematical Sciences  
**Year:** 2023–2025

---

## 📚 References

Key references include:
- Katulevskiy (2024) — ML pipeline for layoff prediction using Layoffs.fyi + Crunchbase data
- Vale (2023) — Epidemiological modeling of the 2022–23 tech layoff wave
- Mamtani & Malukani (2023) — Random Forest achieving 91.5% accuracy on IBM HR attrition dataset
- Díaz et al. (2023) — Explainable AI (SHAP/LIME) for HR analytics
- Pagano et al. (2022) — Fairness and bias in ML employment models


---

<div align="center">

*Made with 📊 data and ❤️ at JAIN University, Bengaluru*

</div>
