# Diabetes Prediction — Logistic Regression vs. KNN

A comparative machine learning project predicting diabetes status from patient health
measurements, using the **Pima Indians Diabetes Dataset**. Two classifiers — **Logistic
Regression** and **K-Nearest Neighbors (KNN)** — are tuned, evaluated, and compared, with a
deliberate focus on **Recall** over Accuracy, given the asymmetric real-world cost of a missed
diagnosis versus a false alarm.

## Project Overview

- **Goal:** predict whether a patient has diabetes (`Outcome`: 0 = No, 1 = Yes) from 8 health
  measurements.
- **Why Recall matters most:** a missed diabetic case (false negative) goes undiagnosed with
  no safety net, while a healthy patient flagged as diabetic (false positive) just gets a
  cheap, low-risk follow-up A1C test. This asymmetry drives every modeling decision in this
  project — hyperparameter tuning and decision threshold selection are both optimized for
  Recall, not Accuracy.
- **Models compared:** Logistic Regression and KNN, each tuned with `GridSearchCV` and
  independently threshold-optimized on a held-out validation set.

## Dataset

The [Pima Indians Diabetes Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
contains 768 records of female patients with 8 numeric features:

| Column | Description |
|---|---|
| `Pregnancies` | Number of times pregnant |
| `Glucose` | Plasma glucose concentration (mg/dL) |
| `BloodPressure` | Diastolic blood pressure (mm Hg) |
| `SkinThickness` | Triceps skinfold thickness (mm) |
| `Insulin` | 2-Hour serum insulin (mu U/ml) |
| `BMI` | Body mass index |
| `DiabetesPedigreeFunction` | Diabetes likelihood based on family history |
| `Age` | Age of the patient (years) |
| `Outcome` | Target variable (0 = Non-diabetic, 1 = Diabetic) |

Class distribution is moderately imbalanced (~65% Non-Diabetic / ~35% Diabetic).

## Methodology

1. **Data Cleaning** — several columns use `0` to silently encode a missing value (e.g. a 0
   mg/dL glucose reading is physiologically impossible). Each affected column was cleaned with
   a physiologically-informed strategy:
   - `Glucose`, `BloodPressure`, `BMI`: zero → NaN → median imputation
   - `SkinThickness`: capped at 60 (clinical caliper limit), zero → NaN → random value within
     ±10 of the patient's WHO BMI-tier median
   - `Insulin`: zero → NaN → random value within ±30 of the patient's Glucose-tier median
     (reflecting that insulin release is directly triggered by blood glucose)
   - Bounded random imputation (rather than plain median) was used specifically to avoid the
     artificial distribution spike that plain median fill produces.

2. **Exploratory Data Analysis** — univariate and bivariate distributions, a correlation
   heatmap, a Variance Inflation Factor (VIF) check for multicollinearity (all features
   cleared with VIF < 2), and a pairplot of the top predictors.

3. **Preprocessing** — stratified Train (56%) / Validation (14%) / Test (30%) split;
   `StandardScaler` fit on training data only.

4. **Modeling** — both models tuned via `GridSearchCV` (5-fold CV, `scoring='recall'`):
   - Logistic Regression: best params `C=0.1`, `class_weight='balanced'`, `penalty='l2'`
   - KNN: best params `n_neighbors=9`, `weights='distance'`, `metric='euclidean'`

5. **Threshold Optimization** — each model's decision threshold was tuned independently on
   the validation set (scanning Precision/Recall across the full 0–1 range) rather than using
   the default 0.5 cutoff, since 0.5 has no particular connection to this project's stated
   cost tradeoff.
   - Logistic Regression: **0.45**
   - KNN: **0.30**

6. **Final Evaluation** — both models evaluated once on the untouched test set.

## Results

| Metric | Logistic Regression (0.45) | KNN (0.30) |
|---|---|---|
| Recall (Diabetes) | 0.78 | **0.81** |
| Precision (Diabetes) | 0.59 | 0.59 |
| F1 (Diabetes) | 0.67 | **0.68** |
| Accuracy | 0.74 | 0.74 |
| AUC | **0.8374** | 0.8263 |
| Average Precision | **0.7170** | 0.6936 |
| False Negatives (missed cases) | 18 | **15** |

**Key finding:** once KNN's threshold was tuned as aggressively toward Recall as Logistic
Regression's, KNN actually caught slightly more diabetic cases in practice (Recall 0.81 vs.
0.78), while Logistic Regression retained a modest edge in overall ranking quality (AUC, AP).
Which model is "better" depends on what's being optimized for — this project treats that as
a genuine finding in itself, not a loose end.

## Repository Structure

```
├── Diabetes_Prediction.ipynb        # Full analysis notebook (cleaning, EDA, modeling, evaluation)
├── Diabetes_Prediction_Report.docx  # Written project report
├── README.md                        # This file
```

## Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
statsmodels
```

## Limitations

- Small dataset (768 rows total) limits how much signal any model can learn, especially for
  the minority (Diabetic) class.
- `SkinThickness` and `Insulin` required substantial imputation; a meaningful portion of
  those two columns is synthetic rather than measured.
- The dataset lacks some clinically important predictors (e.g. HbA1c, fasting glucose test
  results), which likely caps how separable the two classes can ever be with this feature set.
- Results are based on a single train/validation/test split; repeating across multiple random
  splits would give a more robust estimate of each model's true performance and variance.
