Robin - 25-11-26 - Exploration of data

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


