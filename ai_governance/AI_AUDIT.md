# AI Audit Report (Catch & Correct)

## Entry 1 — Naive K-Fold Data Leakage in Validation Design
- **Author:** Robin
- **File / Version:** `src/01_eda_and_validation_setup.py`, `src/02_preprocessing_and_baseline.py` (v1.0 -> v1.1)
- **Error Detected:** Standard random splitting (`train_test_split` / `KFold`) splits frequency sweeps of the exact same aerodynamic configuration across train and validation sets.
- **Root Cause:** The dataset contains 1,503 rows grouped into 106 unique aerodynamic configurations (8 to 19 frequency measurements per configuration). Random splitting causes 100.00% group overlap between train and validation folds, underestimating `DecisionTree_Regularized` CV RMSE by 9.60% (3.7242 dB vs 4.1199 dB).
- **Correction Applied:** Replaced random splits with `GroupShuffleSplit(test_size=0.20, random_state=42)` and `GroupKFold(n_splits=5)` grouped on the 4 aerodynamic parameters, with an explicit programmatic check (`assert set(groups_train).isdisjoint(set(groups_test))`) guaranteeing 0.00% leakage.

---

## Entry 2 — Ineffective `log1p` Preprocessing on Small-Scale Skewed Feature
- **Author:** Robin
- **File / Version:** `src/02_preprocessing_and_baseline.py` (v1.0 -> v1.1)
- **Error Detected:** Applying `FunctionTransformer(np.log1p)` to `suction_side_displacement_thickness` and `frequency` degraded linear model performance (`OLS_LogScaled` Test RMSE = 5.3163 dB vs `OLS_Raw` Test RMSE = 4.9780 dB).
- **Root Cause:** `suction_side_displacement_thickness` takes very small values (0.0004 m to 0.058 m), where $\ln(1 + x) \approx x$, so `log1p` fails to correct its skewness (1.70). Meanwhile, log-transforming `frequency` turns the acoustic spectrum into a non-monotonic bell curve, which a linear regression cannot fit.
- **Correction Applied:** Documented the failure of `log1p` for linear models; planned replacement with `np.log` (valid since all values are strictly positive) or `PowerTransformer(method="yeo-johnson")` inside the `Pipeline`, paired with non-linear / polynomial models in Step 3.

---

## Entry 3 — Hardcoded "Best Linear Baseline" in Residual Diagnostic Plot
- **Author:** Robin
- **File / Version:** `src/02_preprocessing_and_baseline.py` (v1.0 -> v1.1)
- **Error Detected:** Figure `06_baseline_residuals.png` was titled "Best Linear Baseline vs Regularized Tree" and plotted `Ridge_LogScaled` (CV Val RMSE = 4.9341 dB, Test RMSE = 5.3164 dB), whereas `OLS_Raw` was actually the best linear baseline (CV Val RMSE = 4.8134 dB, Test RMSE = 4.9780 dB).
- **Root Cause:** The AI hardcoded `("Ridge_LogScaled", ...)` inside `plot_holdout_residuals()` instead of selecting the best linear pipeline programmatically from the CV results table.
- **Correction Applied:** Flagged the mismatch during manual audit of `baseline_results.csv` and updated the diagnostic selection logic to pick the lowest-CV-RMSE linear model dynamically (or label the plotted model explicitly without claiming it is the best).

---

## Entry 4 — [To be completed in Step 3 / Modeling & Tuning]
- **Author:** Robin
- **File / Version:** `src/03_...` (v1.0 -> v1.1)
- **Error Detected:** [Describe the statistical or ML bug caught in Step 3]
- **Root Cause:** [Explain why the AI produced this error]
- **Correction Applied:** [Describe the code fix and verification]