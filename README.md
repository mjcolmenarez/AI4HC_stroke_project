# Stroke Risk Prediction: Prioritising Adults for Preventive Stroke-Risk Review

This project builds a calibrated stroke-risk classifier and applies the Responsible AI (RAI) Toolbox to move beyond raw accuracy toward a model that is fair, interpretable, and clinically actionable. It was developed for the AI for Healthcare (AI4HC) course as a demonstration of an end-to-end, human-in-the-loop machine learning workflow.

- **Goal:** identify adults at elevated stroke risk from routinely recorded demographic and clinical data, so they can be prioritised for a preventive stroke-risk review — not to diagnose stroke.
- **Notebook:** `notebooks/main_notebook.ipynb`
- **Data:** `data/healthcare-dataset-stroke-data.csv` (Kaggle Stroke Prediction Dataset, 5,110 patient-level records)

**Table of Contents**
- Introduction
- Project Structure
- Install
- Quickstart: Run the Notebook
- Data Description
- Evaluation and Thresholding
- Responsible AI and Reproducibility
- Troubleshooting
- License and Acknowledgements

---

## Introduction

**Purpose.** Among adults seen in outpatient/primary care, can routinely recorded information (age, comorbidities, glucose, BMI, smoking, social variables) identify who is at highest risk of stroke, so that they are invited first for a preventive stroke-risk review (blood-pressure check, HbA1c and lipids, lifestyle counselling)?

**Decision supported.** One yes/no per patient: *invite for a nurse-led review now, or continue usual care.* The model **prioritises**; it never diagnoses and never removes anyone from care. A clinician makes the final call.

**Methods.** A scikit-learn pipeline (preprocessing + logistic regression), probability calibration (Platt scaling), threshold selection compared across three clinically motivated methods plus a statistical reference, one locked final test evaluation, and the Microsoft Responsible AI Toolbox (interpretability, error analysis, counterfactuals, causal inference) for interrogating model behaviour beyond aggregate metrics.

**Interpretation discipline.** Every major analytical stage in the notebook includes a plain-language "So what?" cell translating the result into a clinical or policy implication — coefficients and causal effects describe associations/estimated effects, not proven causal mechanisms, unless explicitly qualified.

**Intended use and limitations.** This is a retrospective, cross-sectional teaching analysis on a single, undocumented data source with no timestamps or site information. It is **not validated for prospective clinical prediction, diagnosis, or deployment**. See "Responsible AI and Reproducibility" below for the full limitations discussion.

---

## Project Structure

```
AI4HC_stroke_project/
├── data/
│   └── healthcare-dataset-stroke-data.csv     Raw dataset (Kaggle, 5,110 records)
├── notebooks/
│   ├── main_notebook.ipynb                    Graded deliverable: all 9 steps + RAI Toolbox
│   └── ML4HL_Stroke_EDA.ipynb                 Original exploratory notebook (historical reference
│                                               only — its findings and decisions are reproduced
│                                               and carried forward into main_notebook.ipynb)
├── utils.py                                   Reusable helper functions (see note below)
├── environment.yml                            Conda environment for reproducibility
├── artifacts/
│   └── locked_threshold.json                  Decision threshold, written before the test set is
│                                               touched (evidence of the locking discipline in
│                                               Section 9.1 of the notebook)
└── README.md                                  This file
```

**A note on `utils.py`:** reusable functions (scoring, threshold-selection methods, subgroup reporting, causal-effect tables, RAI model wrappers) were consolidated directly into the notebook (Section 5.1) for a fully self-contained, single-file analysis that is easy to read top to bottom. `utils.py` reflects an earlier iteration of the project and is kept for reference; the notebook does not currently import from it.

---

## Install

**Prerequisites:** conda (or mamba), Git, and a working Jupyter setup.

**Create the environment:**
```bash
conda env create -f environment.yml
conda activate stroke_rai
```

**Verify Jupyter and widgets:**
```bash
python -c "import IPython, ipywidgets; print('Jupyter OK')"
```
If using JupyterLab and widgets don't render, ensure JupyterLab ≥ 3.x. No manual widget install should be required with this environment.

**Note on the Responsible AI Toolbox stack:** `responsibleai`, `raiwidgets`, `dice-ml`, `econml`, and `fairlearn` are pinned via `environment.yml` for local/conda use. If installing in Google Colab instead, use:
```python
!pip install -q responsibleai==0.36.0 raiwidgets==0.36.0 scipy==1.13.1
```

---

## Quickstart: Run the Notebook

1. Launch Jupyter (`jupyter lab` or open in VS Code) with the `stroke_rai` environment/kernel selected.
2. Open `notebooks/main_notebook.ipynb` and run cells top to bottom.
3. **What the notebook does, in order:**
   - States the clinical question, target population, decision context, and success criteria before any modelling (Section 0).
   - Loads and cleans the raw data, explores distributions, missingness, and bivariate/multivariate relationships to the outcome (Sections 1–4).
   - Restricts the modelling cohort to adults 18+ and excludes a single unusable gender category (Section 6), then splits into a 70/15/15 stratified train/validation/test set (Section 7).
   - Builds a leakage-checked preprocessing pipeline and a baseline logistic regression model (Section 7.1–7.2), and checks discrimination by subgroup against an age-only benchmark (Section 7.3).
   - Calibrates the model's probabilities (Section 8) and selects a decision threshold on validation only, comparing three clinical methods and one statistical reference (Section 9).
   - Locks the threshold to disk **before** touching the test set, then reports the one and only final test evaluation, including a subgroup fairness check (Sections 9.1–10.1).
   - Runs the Responsible AI Toolbox — data analysis, model overview & fairness, error analysis, feature importance, counterfactuals, and causal analysis — and launches two interactive dashboards (Section 11).
   - Closes with a policy-implications table (pros, cons, consequences, implications) and a reproducibility statement (Section 12).

---

## Data Description

- **Source:** Kaggle Stroke Prediction Dataset (`fedesoriano`). Confidential/undocumented source per the dataset's own description; used for educational purposes only.
- **Rows:** 5,110 patient-level records. **Modelling cohort:** 4,253 adults (18+) after excluding one `gender = 'Other'` record and 856 patients under 18.
- **Target:** `stroke` (1 = stroke recorded, 0 = not). Prevalence ≈ 4.9% overall, ≈ 5.8% in the adult modelling cohort.
- **Features used:** `age`, `avg_glucose_level`, `bmi`, `hypertension`, `heart_disease`, `gender`, `ever_married`, `work_type`, `Residence_type`, `smoking_status`.
- **Key caveats:**
  - `bmi` has ~4% missing values, imputed with the **training-set median** inside the pipeline (never on the full dataset, to avoid leakage).
  - `smoking_status = 'Unknown'` (≈30% of records) is disguised missingness, not a genuine category — kept separate rather than merged, since younger patients are disproportionately "Unknown," and collapsing it produces a Simpson's-paradox-style misreading of risk.
  - `work_type` and `ever_married` are largely proxies for age (confirmed via age-adjusted odds ratios and later via RAI feature importance) and are deliberately retained in the model to test and demonstrate this proxy effect, rather than removed upfront.
  - The outcome is "stroke recorded", not a clinically confirmed incident stroke with a defined follow-up window — some records may reflect prior, not incident, events.
  - Single, undocumented source with no dates or sites: representativeness and external validity cannot be verified, and any performance reported here is an internal estimate only.

---

## Evaluation and Thresholding

- **Model:** logistic regression inside a `scikit-learn` pipeline (median imputation + scaling for numeric features, most-frequent imputation for binary features, most-frequent imputation + one-hot encoding for categorical features), fit on training data only.
- **Baseline:** a majority-class (`DummyClassifier`) baseline is reported first; a model must clearly beat this to justify use, and an age-only benchmark is reported alongside it throughout, since age dominates the model's discrimination.
- **Calibration:** `CalibratedClassifierCV` (5-fold, Platt/sigmoid scaling), fit on training data only, evaluated on validation. Reliability assessed via calibration-in-the-large, quantile-binned reliability tables/diagrams, and a calibration slope.
- **Threshold selection:** chosen on the **validation set only**, comparing a recall-floor method (≥80% sensitivity), a workload-cap method (≤20% referral rate), a cost-based method (a missed stroke weighted 20× a false alarm), and Youden's J as a statistical reference. The cost-based threshold (≈0.048) was locked, since it converges with the recall-floor result while being more stable under resampling.
- **Locking discipline:** the threshold is written to `artifacts/locked_threshold.json` before the test set is used for anything. The test set is read exactly once for final evaluation (Section 10).
- **Final test result:** sensitivity ≈76% (28 of 37 strokes caught), referring ≈35% of adults, with subgroup and fairness checks reported at the locked threshold (Section 10.1) — including an identified sensitivity gap between men and women aged 50–64 that is flagged as a monitoring priority, not resolved in this version.

---

## Responsible AI and Reproducibility

**Interpretability.** Global feature importance is reported via the RAI Toolbox's explainer; age accounts for roughly two-thirds of the model's importance, followed by glucose, work type (a suspected age proxy, confirmed via ablation), hypertension, and smoking status.

**Error analysis.** The RAI error-analysis tree separates errors by type rather than reporting a single blended error rate: most errors among older patients are false alarms (expected, and low-harm under the stated cost ratio), while nearly all missed strokes occur in a smaller, younger, higher-glucose subgroup — the clinically important error to monitor.

**Counterfactuals.** Only clinically modifiable features (`avg_glucose_level`, `hypertension`) are permitted to vary; age, sex, and BMI are frozen, since they are either non-modifiable or (in BMI's case) causally unreliable in this data. Roughly 61% of referred 50–64-year-olds could drop below the referral threshold with a realistic reduction in glucose and/or controlled hypertension; referred patients aged 65+ are referred almost entirely on the basis of age and cannot be moved below the threshold this way.

**Causal analysis.** Uses EconML's double machine learning, treating `avg_glucose_level`, `bmi`, `hypertension`, `heart_disease`, and `smoking_status` as candidate modifiable treatments, with `ever_married` included as a negative control (expected to show no real effect once age is accounted for). Hypertension shows the clearest modifiable effect on stroke risk; the estimated BMI effect is treated as unreliable (likely reverse causation in cross-sectional data, not a genuine protective effect) rather than reported at face value. Causal estimates should be treated as hypothesis-generating: confidence intervals can shift modestly between reruns due to the underlying method's internal cross-fitting, and no causal claim here should be read as clinically confirmatory without prospective, dated data.

**Governance and equity.** A quarterly monitoring plan is proposed for the identified sensitivity gap (women aged 50–64), the reliance on `work_type` as an age proxy, and general calibration drift, with specific alert triggers documented in the notebook (Section 11.8).

**Reproducibility.** `RANDOM_STATE = 42` fixes splits, model fitting, calibration folds, and bootstrap resampling throughout. All preprocessing is fit inside the pipeline on training data only (verified explicitly in Section 7.2.1's leakage check). Calibration and thresholding use training/validation only; the test set is touched exactly once. Library versions are printed at the end of the notebook and pinned in `environment.yml`.

**License and data stewardship.** This is a synthetic/undocumented teaching dataset. Any real-world use of a model like this would require institutional governance, a privacy/bias audit, and likely qualifies as high-risk clinical decision support under frameworks such as the EU AI Act — none of which has been performed here.

---

## Troubleshooting

- **RAI dashboard not rendering:** trust the notebook (File → Trust Notebook) and prefer JupyterLab ≥ 3.x over classic Notebook. The dashboard opens a local link (e.g. `http://localhost:...`) — open it in a browser tab and keep the notebook kernel running in the background.
- **Import errors** (e.g. `fairlearn`, `responsibleai`, `raiwidgets`): recreate the environment — `conda env remove -n stroke_rai && conda env create -f environment.yml`.
- **File not found on `DATA_PATH`:** confirm you are running the notebook from within `notebooks/`, so that the relative path `../data/healthcare-dataset-stroke-data.csv` resolves correctly.
- **Causal analysis numbers differ slightly between runs:** this is expected — EconML's double machine learning involves internal cross-fitting with some run-to-run variation. Directional conclusions (which effects clear zero) are stable; exact point estimates may shift marginally.

---

## License and Acknowledgements

- **Dataset:** Kaggle Stroke Prediction Dataset (`fedesoriano`), used for educational purposes only.
- **Methodology:** adapted from the AI4HC course's Responsible AI Toolbox teaching materials, based on Microsoft's Responsible AI Toolbox and standard scikit-learn practice.
- **License:** see `LICENSE`.