#  Diabetes Pre-Screening Classifier

A machine learning pipeline that flags individuals at elevated diabetes risk using low-cost clinical measurements — without requiring formal lab tests. Built on the Pima Indians Diabetes dataset (NIDDK), this project is framed as a **pre-screening tool** for primary care, tele-health, and insurance intake contexts where full diagnostic testing is unavailable or cost-prohibitive.

---

## Problem Statement

Diabetes often goes undetected until complications arise. Clinical diagnosis requires fasting glucose tests, HbA1c measurements, or oral glucose tolerance tests — all of which need lab infrastructure and add cost. This classifier uses 8 easily collectable patient attributes to estimate diabetes risk and prioritise who needs formal testing.

**The core constraint driving every modelling decision:** a missed diabetic patient (false negative) is far more dangerous than an unnecessary follow-up test (false positive). The entire pipeline — class weighting, metric selection, threshold optimisation — is built around minimising false negatives.

---

## Dataset

**Source:** [Pima Indians Diabetes Database — Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) 
**Population:** Adult women of Pima Indian heritage, age ≥ 21

**Size:** 768 records, 8 features, binary outcome

| Feature | Description |
|---|---|
| Pregnancies | Number of times pregnant |
| Glucose | Plasma glucose concentration (2-hr oral glucose tolerance test) |
| BloodPressure | Diastolic blood pressure (mm Hg) |
| SkinThickness | Triceps skinfold thickness (mm) |
| Insulin | 2-hour serum insulin (μU/mL) |
| BMI | Body mass index (kg/m²) |
| DiabetesPedigreeFunction | Genetic risk score based on family history |
| Age | Age in years |

**Target:** `Outcome` — 1 = diabetic, 0 = non-diabetic (65/35 split)

> Download the CSV and place it at `data/diabetes.csv` before running the notebook.

---

## Project Structure

```
diabetes-prescreening-classifier/
│
├── data/
│   └── README.md              # Download instructions for dataset
│
├── Diabetes.ipynb             # Main analysis notebook          
├── .gitignore                 
├── LICENSE                    
└── README.md                  
```

---

## Methodology

### 1. Data Cleaning
- Zeroes in `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, and `BMI` are biologically impossible — replaced with `NaN` and imputed
- **Mean imputation** for near-symmetric features (Glucose, BloodPressure)
- **Median imputation** for right-skewed features (SkinThickness, Insulin, BMI) — median is robust to the long tails present in these distributions
- Insulin had ~49% missing values after zero removal — flagged as a reliability concern in the analysis

### 2. Exploratory Data Analysis
- Scatter plots of Age and Pregnancies vs clinical features, coloured by Outcome, to identify visual separation between diabetic and non-diabetic patients
- Correlation heatmap to identify the strongest predictors and potential multicollinearity pairs
- Class distribution check to quantify imbalance and inform model configuration

### 3. Modelling
- **Train/test split:** 80/20, `random_state=2`
- **Scaling:** `StandardScaler` fit on training set only — applied to test set without refitting to prevent data leakage
- **Baseline comparison** across 5 models: Logistic Regression, SVM, LightGBM, Random Forest, KNN
- **Class imbalance handling:** `class_weight='balanced'` for LR/SVM; `scale_pos_weight` for LightGBM

### 4. Hyperparameter Tuning
- **Framework:** [Optuna](https://optuna.org/) with TPE sampler (Bayesian optimisation)
- **Objective:** F2 score via 5-fold cross-validation — β=2 means recall is weighted twice as heavily as precision, directly encoding the clinical cost asymmetry
- **Models tuned:** Logistic Regression, SVM, LightGBM (50 trials each)

### 5. Threshold Optimisation
- Default 0.5 probability threshold is not optimal for screening — replaced by sweeping the precision-recall curve to find the threshold that maximises F2 score
- Lower threshold = more patients flagged = higher recall at the cost of more false positives (acceptable for a pre-screening tool)

---

## Results

### Baseline Model Comparison

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| SVM | 0.753 | 0.556 | **0.778** | 0.648 | **0.829** |
| Logistic Regression | 0.753 | 0.561 | 0.711 | 0.627 | 0.805 |
| Random Forest | 0.734 | 0.545 | 0.533 | 0.539 | 0.816 |
| LightGBM | 0.714 | 0.509 | 0.644 | 0.569 | 0.797 |
| KNN | 0.721 | 0.521 | 0.556 | 0.538 | 0.791 |

SVM leads on recall and ROC-AUC at baseline. LightGBM underperforms before tuning — expected, given its sensitivity to hyperparameter configuration.

> Tuned model results and threshold-optimised metrics are in the notebook.

---

## Key Design Decisions

| Decision | Choice | Why |
|---|---|---|
| Primary metric | F2 Score | Penalises false negatives 4× more than false positives |
| Imputation strategy | Mean / Median (feature-specific) | Transparent, leakage-free, auditable in clinical context |
| Tuning framework | Optuna TPE | More efficient than grid/random search on multi-parameter spaces |
| Threshold | F2-optimal from PR curve | Avoids arbitrary 0.5 default that ignores cost asymmetry |
| Scaling | StandardScaler (train-only fit) | Prevents leakage; required for LR and SVM |

---

## Setup

```bash
git clone https://github.com/Hitesh7234/diabetes-prescreening-classifier.git
cd diabetes-prescreening-classifier
pip install -r requirements.txt
```

Download the dataset from [Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) and place it at `data/diabetes.csv`.

Then open `Diabetes.ipynb` in Jupyter or Google Colab.

---

## Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
lightgbm
xgboost
optuna
```

---

## To-Do

- [ ] Add SHAP values for per-patient explainability — currently feature importance is split-count based (LightGBM), which can misrepresent contribution magnitude
- [ ] Calibration plot to verify that predicted probabilities are well-calibrated (a predicted 70% risk should correspond to ~70% actual prevalence)
- [ ] Experiment with SMOTE or other oversampling techniques as an alternative to class weighting — compare recall vs precision trade-off
- [ ] Build a simple Streamlit app where a user inputs clinical values and gets a risk flag + probability score
- [ ] Validate on an external cohort to test generalisability beyond the Pima population
- [ ] Add early stopping to LightGBM training to find the optimal number of boosting rounds automatically
- [ ] Explore MICE (Multiple Imputation by Chained Equations) for Insulin, given its ~49% missing rate, and compare against median imputation

---

## Author

**Hitesh**  
[LinkedIn](https://linkedin.com/in/yourprofile](https://www.linkedin.com/in/hiteshsinghstats/) · [GitHub](https://github.com/Hitesh7234)
