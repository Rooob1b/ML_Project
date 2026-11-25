Robin - 25/11/26 - Exploration of data

Target file: src/01_eda_and_validation_setup.py (do not modify any other files).
Context: Strictly follow @PROJECT_GUIDELINES.md and @GEMINI.md. Do NOT read data/airfoil_self_noise.dat raw into context; inspect it only through the Python script.

Task: Implement a modular, deterministic (RANDOM_SEED = 42) EDA and validation-audit script for the NASA Airfoil Self-Noise continuous regression dataset, strictly following the workflow (Understand -> Clean -> Split -> Transform -> Learn -> Evaluate). Do NOT fit any scalers, imputers, or models in this script.

1. Data Loading & Physical Plausibility Audit (Accuracy & Completeness):
- Resolve project root dynamically using pathlib.Path(__file__).resolve().parents[1] so the script runs from any working directory.
- Load data/airfoil_self_noise.dat (whitespace-delimited, no header) with exact schema: ['frequency', 'angle_of_attack', 'chord_length', 'free_stream_velocity', 'suction_side_displacement_thickness', 'scaled_sound_pressure_level'].
- Assert shape (1503, 6), check dtypes, missing values, and exact duplicates.
- Run physical domain sanity checks to separate impossible values from valid extremes: verify frequency > 0, chord_length > 0, free_stream_velocity > 0, suction_side_displacement_thickness > 0, and angle_of_attack >= 0. Log pass/fail counts.

2. Statistical Diagnostics, Non-Linear Relevance & Explicit Fallbacks:
- Compute descriptive stats (mean, std, median, IQR, skewness, kurtosis) and export a clean summary table to report/tables/eda_summary.csv for LaTeX reuse.
- Run Shapiro-Wilk normality tests (alpha = 0.05) on all 6 variables. Fallback rule: if p < 0.05 for any variable, explicitly log the violation, use Spearman rank correlation as the primary monotonic metric (report Pearson only for contrast), and flag features with |skewness| > 1.0 as candidates for inside-Pipeline log/power transforms.
- Compute Mutual Information (sklearn.feature_selection.mutual_info_regression, random_state=42) between the 5 features and the target as a filter-method diagnostic to capture non-linear / non-monotonic dependencies (e.g., frequency vs SPL) that correlation misses. Print a ranked comparison table (Pearson vs Spearman vs Mutual Information) and note explicitly that any actual feature selection must happen inside CV folds during modeling.
- Compute VIF (statsmodels.stats.outliers_influence.variance_inflation_factor) on the 5 features to diagnose multicollinearity (especially angle_of_attack vs suction_side_displacement_thickness).
- Quantify statistical outliers via the 1.5 * IQR rule per column and check target tail coverage (deciles / extreme dB regimes) and sample balance across discrete experimental levels (chord_length, free_stream_velocity). Constraint: report counts and physical ranges only — do NOT drop any rows (valid extreme aerodynamic regimes must be preserved).

3. Experimental Grouping & Data Leakage Audit:
- Define aerodynamic configuration groups using the 4-tuple ['angle_of_attack', 'chord_length', 'free_stream_velocity', 'suction_side_displacement_thickness']. Print total unique groups and min/median/max sweep lengths.
- Quantify leakage: compare KFold(n_splits=5, shuffle=True, random_state=42) vs GroupKFold(n_splits=5) by printing, for each fold, the percentage of validation rows whose aerodynamic configuration also appears in the training fold (expected: ~100% for KFold vs 0% for GroupKFold).

4. Publication-Ready Figures (saved to report/figures/, 300 DPI, plt.close() after save, no plt.show()):
- 01_distributions.png: Histograms + KDE of all 6 variables + log10(frequency) with physical units on axes.
- 02_spearman_correlation.png: Annotated lower-triangle Spearman correlation heatmap.
- 03_feature_relevance.png: Side-by-side horizontal bar chart comparing |Spearman rho| vs Mutual Information score for each feature against scaled_sound_pressure_level.
- 04_acoustic_spectra.png: scaled_sound_pressure_level (dB) vs frequency (Hz, log x-axis) colored by angle_of_attack for a fixed chord length (0.3048 m) and velocity (71.3 m/s).

5. Code Quality Constraints:
- Decompose into small, single-responsibility functions with Python type hints and concise docstrings, orchestrated under if __name__ == "__main__":.
- Create output directories (report/figures, report/tables) automatically with mkdir(parents=True, exist_ok=True).



Robin - 25/11/26 - Preprocessing and Baseline
Act as a senior Machine Learning engineer working on the NASA Airfoil Self-Noise regression project (supervised regression, target: scaled_sound_pressure_level in dB).

Create the script `src/02_preprocessing_and_baseline.py` following strict anti-leakage and reproducibility engineering standards.

### 1. Strict Guardrails & Constraints
- Set `RANDOM_SEED = 42` for Python, NumPy, and all stochastic estimators/splitters.
- Resolve paths dynamically from `Path(__file__).resolve().parents[1]` (must run cleanly from repo root or `src/`).
- Load `data/airfoil_self_noise.dat` (1503 rows x 6 columns, whitespace-separated, no header) using the exact schema from `01_eda_and_validation_setup.py`:
  `["frequency", "angle_of_attack", "chord_length", "free_stream_velocity", "suction_side_displacement_thickness", "scaled_sound_pressure_level"]`.
- Define aerodynamic configuration `groups` by grouping on the 4 non-frequency features:
  `["angle_of_attack", "chord_length", "free_stream_velocity", "suction_side_displacement_thickness"]` (106 unique groups).
- NEVER fit any transformer, scaler, or model on the full dataset or on the test set. Every preprocessing step MUST live inside a `sklearn.pipeline.Pipeline` / `ColumnTransformer`.
- Do NOT drop any rows or outliers (all 1,503 rows are physically valid).

### 2. Data Splitting & Leakage Verification
1. Hold-out split: Use `GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)` on `(X, y, groups)` to create `X_train, X_test, y_train, y_test, groups_train, groups_test`.
2. Assert programmatically that `set(groups_train).isdisjoint(set(groups_test))` (0% group overlap) and print train/test row counts and unique group counts.
3. Cross-validation protocol on `X_train` only: Use `GroupKFold(n_splits=5)` passed with `groups=groups_train`.
4. Leave `X_test, y_test` completely untouched during CV model comparison; only evaluate the final selected baseline pipelines on `X_test` once at the very end for an unbiased hold-out sanity check (clearly labeled as "Hold-out Test").

### 3. Preprocessing Pipelines (Justified by Step 1 EDA)
Build two reusable `ColumnTransformer` preprocessors:
- `prep_scale_only`: `StandardScaler()` on all 5 numeric features.
- `prep_log_and_scale`: Apply `FunctionTransformer(np.log1p, validate=False)` followed by `StandardScaler()` to the two highly skewed features (`frequency`, `suction_side_displacement_thickness`), and `StandardScaler()` to the remaining 3 features (`angle_of_attack`, `chord_length`, `free_stream_velocity`).

### 4. Baseline Models to Benchmark (5-Fold GroupKFold)
Evaluate the following 6 pipelines using `cross_validate(..., cv=GroupKFold(n_splits=5), groups=groups_train, return_train_score=True)` with `RMSE`, `MAE`, and `R2`:
1. `Dummy_Mean`: `DummyRegressor(strategy="mean")`
2. `OLS_Raw`: `LinearRegression()` (no scaling, no log transform)
3. `OLS_LogScaled`: `prep_log_and_scale` + `LinearRegression()`
4. `Ridge_LogScaled`: `prep_log_and_scale` + `Ridge(alpha=1.0, random_state=42)`
5. `DecisionTree_Unconstrained`: `DecisionTreeRegressor(random_state=42)` (to explicitly illustrate high variance / overfitting)
6. `DecisionTree_Regularized`: `DecisionTreeRegressor(max_depth=6, min_samples_leaf=5, random_state=42)` (controlled complexity baseline)

Also run a brief leakage-contrast check on `OLS_LogScaled` and `DecisionTree_Regularized` comparing `GroupKFold(5)` vs naive `KFold(5, shuffle=True, random_state=42)` CV RMSE on `X_train`, and print the delta to quantify how much naive KFold underestimates generalization error.

### 5. Outputs & Publication-Ready Artifacts
1. Print a clean comparison table with: `model`, `cv_train_rmse`, `cv_val_rmse`, `cv_val_rmse_std`, `cv_val_mae`, `cv_val_r2`, `overfit_gap_rmse` (`cv_val_rmse - cv_train_rmse`), and `test_rmse`, `test_mae`, `test_r2`.
2. Export the full summary table to `report/tables/baseline_results.csv`.
3. Generate and save 2 figures (300 DPI, call `plt.close()` after each) in `report/figures/`:
   - `05_baseline_train_vs_val_rmse.png`: Grouped horizontal bar chart comparing CV Train RMSE vs CV Validation RMSE (with ±1 SD error bars on validation) for all 6 baseline models, clearly highlighting underfitting vs overfitting.
   - `06_baseline_residuals.png`: Side-by-side diagnostic plot (Predicted vs Actual dB, and Residuals vs Predicted dB) on the hold-out test set for the best linear baseline (`Ridge_LogScaled`) vs `DecisionTree_Regularized`.

Run `python src/02_preprocessing_and_baseline.py` to verify exit code 0 and confirm all tables and figures are created cleanly.