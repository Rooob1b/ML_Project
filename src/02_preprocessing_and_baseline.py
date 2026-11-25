"""Preprocessing pipelines and baseline model benchmarking for NASA Airfoil Self-Noise.

Workflow stage: Group-aware split -> Pipeline encapsulation -> 5-fold GroupKFold CV
-> Leakage contrast (GroupKFold vs naive KFold) -> Single hold-out test check.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.tree import DecisionTreeRegressor

RANDOM_SEED: int = 42

SCHEMA_COLUMNS: List[str] = [
    "frequency",
    "angle_of_attack",
    "chord_length",
    "free_stream_velocity",
    "suction_side_displacement_thickness",
    "scaled_sound_pressure_level",
]
FEATURE_COLUMNS: List[str] = SCHEMA_COLUMNS[:-1]
TARGET_COLUMN: str = SCHEMA_COLUMNS[-1]
GROUP_COLUMNS: List[str] = [
    "angle_of_attack",
    "chord_length",
    "free_stream_velocity",
    "suction_side_displacement_thickness",
]

SKEWED_FEATURES: List[str] = [
    "frequency",
    "suction_side_displacement_thickness",
]
OTHER_FEATURES: List[str] = [
    "angle_of_attack",
    "chord_length",
    "free_stream_velocity",
]


def set_reproducibility(seed: int = RANDOM_SEED) -> None:
    """Set deterministic random seeds for Python and NumPy."""
    random.seed(seed)
    np.random.seed(seed)


def resolve_paths() -> Tuple[Path, Path, Path, Path]:
    """Resolve project root, dataset path, figures directory, and tables directory."""
    project_root = Path(__file__).resolve().parents[1]
    data_path = project_root / "data" / "airfoil_self_noise.dat"
    figures_dir = project_root / "report" / "figures"
    tables_dir = project_root / "report" / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    return project_root, data_path, figures_dir, tables_dir


def load_dataset_and_groups(
    data_path: Path,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Load raw dataset (no rows dropped) and construct aerodynamic configuration groups."""
    df = pd.read_csv(data_path, sep=r"\s+", header=None, names=SCHEMA_COLUMNS)
    assert df.shape == (1503, 6), f"Expected shape (1503, 6), got {df.shape}"
    assert int(df.isna().sum().sum()) == 0, "Unexpected missing values in raw dataset."

    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()
    groups = df.groupby(GROUP_COLUMNS, sort=False).ngroup()
    assert groups.nunique() == 106, f"Expected 106 unique groups, got {groups.nunique()}"
    return X, y, groups


def split_holdout_by_group(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    test_size: float = 0.20,
    seed: int = RANDOM_SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
    """Perform a leakage-free group-aware train/test hold-out split."""
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))

    X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
    y_train, y_test = y.iloc[train_idx].copy(), y.iloc[test_idx].copy()
    groups_train = groups.iloc[train_idx].copy()
    groups_test = groups.iloc[test_idx].copy()

    assert set(groups_train).isdisjoint(set(groups_test)), (
        "Data leakage detected: train and test groups overlap!"
    )

    print("=" * 78)
    print("1. GROUP-AWARE HOLD-OUT SPLIT & LEAKAGE VERIFICATION")
    print("=" * 78)
    print(
        f"Train set: {len(X_train)} rows ({100.0 * len(X_train) / len(X):.1f}%), "
        f"{groups_train.nunique()} unique aerodynamic groups"
    )
    print(
        f"Test set : {len(X_test)} rows ({100.0 * len(X_test) / len(X):.1f}%), "
        f"{groups_test.nunique()} unique aerodynamic groups"
    )
    print("Group overlap check: set(groups_train).isdisjoint(set(groups_test)) -> PASS (0 overlap)")

    return X_train, X_test, y_train, y_test, groups_train, groups_test


def build_preprocessors() -> Tuple[ColumnTransformer, ColumnTransformer]:
    """Construct unfitted ColumnTransformer preprocessors justified by Step 1 EDA."""
    prep_scale_only = ColumnTransformer(
        transformers=[
            ("scale_all", StandardScaler(), FEATURE_COLUMNS),
        ],
        remainder="drop",
    )

    log_then_scale = Pipeline(
        steps=[
            ("log1p", FunctionTransformer(np.log1p, validate=False)),
            ("scaler", StandardScaler()),
        ]
    )
    prep_log_and_scale = ColumnTransformer(
        transformers=[
            ("log_scale_skewed", log_then_scale, SKEWED_FEATURES),
            ("scale_other", StandardScaler(), OTHER_FEATURES),
        ],
        remainder="drop",
    )
    return prep_scale_only, prep_log_and_scale


def build_baseline_pipelines(
    prep_log_and_scale: ColumnTransformer,
) -> Dict[str, Pipeline]:
    """Define the 6 baseline model pipelines with strict Pipeline encapsulation."""
    return {
        "Dummy_Mean": Pipeline(
            steps=[("model", DummyRegressor(strategy="mean"))]
        ),
        "OLS_Raw": Pipeline(
            steps=[("model", LinearRegression())]
        ),
        "OLS_LogScaled": Pipeline(
            steps=[
                ("prep", clone(prep_log_and_scale)),
                ("model", LinearRegression()),
            ]
        ),
        "Ridge_LogScaled": Pipeline(
            steps=[
                ("prep", clone(prep_log_and_scale)),
                ("model", Ridge(alpha=1.0, random_state=RANDOM_SEED)),
            ]
        ),
        "DecisionTree_Unconstrained": Pipeline(
            steps=[("model", DecisionTreeRegressor(random_state=RANDOM_SEED))]
        ),
        "DecisionTree_Regularized": Pipeline(
            steps=[
                (
                    "model",
                    DecisionTreeRegressor(
                        max_depth=6,
                        min_samples_leaf=5,
                        random_state=RANDOM_SEED,
                    ),
                )
            ]
        ),
    }


def evaluate_cv_baselines(
    pipelines: Dict[str, Pipeline],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    groups_train: pd.Series,
    n_splits: int = 5,
) -> pd.DataFrame:
    """Benchmark all pipelines using 5-fold GroupKFold CV on X_train only."""
    cv = GroupKFold(n_splits=n_splits)
    scoring = {
        "rmse": "neg_root_mean_squared_error",
        "mae": "neg_mean_absolute_error",
        "r2": "r2",
    }

    records = []
    for name, pipe in pipelines.items():
        res = cross_validate(
            pipe,
            X_train,
            y_train,
            groups=groups_train,
            cv=cv,
            scoring=scoring,
            return_train_score=True,
        )
        train_rmse = max(0.0, float(-res["train_rmse"].mean()))
        val_rmses = -res["test_rmse"]
        val_rmse = float(val_rmses.mean())
        val_rmse_std = float(val_rmses.std(ddof=1))
        val_mae = float(-res["test_mae"].mean())
        val_r2 = float(res["test_r2"].mean())

        records.append({
            "model": name,
            "cv_train_rmse": train_rmse,
            "cv_val_rmse": val_rmse,
            "cv_val_rmse_std": val_rmse_std,
            "cv_val_mae": val_mae,
            "cv_val_r2": val_r2,
            "overfit_gap_rmse": val_rmse - train_rmse,
        })

    return pd.DataFrame(records)


def run_leakage_contrast_check(
    pipelines: Dict[str, Pipeline],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    groups_train: pd.Series,
) -> pd.DataFrame:
    """Compare GroupKFold(5) vs naive KFold(5) CV RMSE to quantify leakage optimism."""
    gkf = GroupKFold(n_splits=5)
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    target_models = ["OLS_LogScaled", "DecisionTree_Regularized"]

    rows = []
    for name in target_models:
        pipe = pipelines[name]
        gkf_res = cross_validate(
            pipe,
            X_train,
            y_train,
            groups=groups_train,
            cv=gkf,
            scoring="neg_root_mean_squared_error",
        )
        kf_res = cross_validate(
            pipe,
            X_train,
            y_train,
            cv=kf,
            scoring="neg_root_mean_squared_error",
        )
        gkf_rmse = float(-gkf_res["test_score"].mean())
        kf_rmse = float(-kf_res["test_score"].mean())
        delta = gkf_rmse - kf_rmse
        pct_underestimate = 100.0 * delta / gkf_rmse
        rows.append({
            "model": name,
            "group_kfold_rmse": gkf_rmse,
            "naive_kfold_rmse": kf_rmse,
            "delta_rmse": delta,
            "underestimate_pct": pct_underestimate,
        })

    contrast_df = pd.DataFrame(rows)
    print("\n" + "=" * 78)
    print("2. LEAKAGE-CONTRAST CHECK (5-Fold GroupKFold vs Naive KFold on X_train)")
    print("=" * 78)
    print(contrast_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(
        "[INTERPRETATION] Positive delta_rmse confirms naive KFold leaks frequency sweeps "
        "across folds and underestimates true out-of-group generalization error."
    )
    return contrast_df


def evaluate_holdout_test_once(
    pipelines: Dict[str, Pipeline],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """Fit each baseline pipeline on X_train and evaluate once on untouched Hold-out Test."""
    test_records = []
    test_predictions: Dict[str, np.ndarray] = {}

    for name, pipe in pipelines.items():
        fitted_pipe = clone(pipe)
        fitted_pipe.fit(X_train, y_train)
        y_pred = fitted_pipe.predict(X_test)
        test_predictions[name] = y_pred

        test_records.append({
            "model": name,
            "test_rmse": float(root_mean_squared_error(y_test, y_pred)),
            "test_mae": float(mean_absolute_error(y_test, y_pred)),
            "test_r2": float(r2_score(y_test, y_pred)),
        })

    return pd.DataFrame(test_records), test_predictions


def audit_test_residuals(
    y_test: pd.Series,
    test_predictions: Dict[str, np.ndarray],
    model_names: List[str],
) -> None:
    """Run Shapiro-Wilk normality and summary checks on Hold-out Test residuals."""
    print("\n" + "=" * 78)
    print("4. HOLD-OUT TEST RESIDUAL DIAGNOSTICS (Linear vs Tree Baseline)")
    print("=" * 78)
    for name in model_names:
        residuals = y_test.to_numpy() - test_predictions[name]
        w_stat, p_val = stats.shapiro(residuals)
        status = "normal" if p_val >= 0.05 else "non-normal"
        print(
            f"  - {name:<26}: mean={residuals.mean():+.4f} dB, "
            f"std={residuals.std(ddof=1):.4f} dB, "
            f"Shapiro-Wilk W={w_stat:.4f}, p={p_val:.3e} ({status})"
        )


def plot_train_vs_val_rmse(results_df: pd.DataFrame, figures_dir: Path) -> Path:
    """Save 05_baseline_train_vs_val_rmse.png comparing CV Train vs Validation RMSE."""
    ordered = results_df.sort_values(by="cv_val_rmse", ascending=False).reset_index(drop=True)
    y_pos = np.arange(len(ordered))
    bar_h = 0.36

    fig, ax = plt.subplots(figsize=(10.5, 6))
    ax.barh(
        y_pos + bar_h / 2,
        ordered["cv_train_rmse"],
        height=bar_h,
        label="CV Train RMSE",
        color="#4c78a8",
        edgecolor="white",
    )
    ax.barh(
        y_pos - bar_h / 2,
        ordered["cv_val_rmse"],
        xerr=ordered["cv_val_rmse_std"],
        height=bar_h,
        label="CV Validation RMSE (±1 SD)",
        color="#e45756",
        edgecolor="white",
        capsize=4,
        error_kw={"ecolor": "#222222", "elinewidth": 1.2},
    )

    for idx, row in ordered.iterrows():
        ax.text(
            row["cv_train_rmse"] + 0.08,
            idx + bar_h / 2,
            f"{row['cv_train_rmse']:.2f}",
            va="center",
            fontsize=8.5,
            color="#2c4a6e",
        )
        ax.text(
            row["cv_val_rmse"] + row["cv_val_rmse_std"] + 0.08,
            idx - bar_h / 2,
            f"{row['cv_val_rmse']:.2f} (±{row['cv_val_rmse_std']:.2f})",
            va="center",
            fontsize=8.5,
            color="#9e2a2b",
        )

    # Highlight underfitting (Dummy_Mean) and severe overfitting (DecisionTree_Unconstrained)
    for idx, row in ordered.iterrows():
        if row["model"] == "Dummy_Mean":
            ax.text(
                row["cv_val_rmse"] * 0.48,
                idx - bar_h / 2,
                "Underfitting (zero-skill baseline)",
                va="center",
                ha="center",
                fontsize=8.5,
                fontweight="bold",
                color="white",
            )
        elif row["model"] == "DecisionTree_Unconstrained":
            ax.annotate(
                f"Severe overfitting (Train = 0.00, gap = {row['overfit_gap_rmse']:.2f} dB)",
                xy=(0.55, idx + bar_h / 2),
                xytext=(1.75, idx + bar_h / 2),
                va="center",
                arrowprops={"arrowstyle": "->", "color": "#1f3a5f", "lw": 1.1},
                fontsize=8.5,
                fontweight="bold",
                color="#1f3a5f",
            )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(ordered["model"], fontsize=9.5)
    ax.set_xlabel("Root Mean Squared Error — RMSE (dB)", fontsize=10.5)
    ax.set_title(
        "5-Fold GroupKFold Baseline Benchmark: CV Train vs Validation RMSE",
        fontsize=12,
        pad=12,
    )
    ax.set_xlim(0, ordered["cv_val_rmse"].max() + ordered["cv_val_rmse_std"].max() + 2.0)
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    ax.legend(loc="upper right", frameon=True)
    fig.tight_layout()

    out_path = figures_dir / "05_baseline_train_vs_val_rmse.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def plot_holdout_residuals(
    y_test: pd.Series,
    test_predictions: Dict[str, np.ndarray],
    results_df: pd.DataFrame,
    figures_dir: Path,
) -> Path:
    """Save 06_baseline_residuals.png: Hold-out Test Predicted vs Actual & Residuals."""
    models_to_plot = [
        ("Ridge_LogScaled", "#2b5c8f"),
        ("DecisionTree_Regularized", "#d97724"),
    ]
    y_true = y_test.to_numpy()
    min_val = float(min(y_true.min(), *(test_predictions[m].min() for m, _ in models_to_plot))) - 1.0
    max_val = float(max(y_true.max(), *(test_predictions[m].max() for m, _ in models_to_plot))) + 1.0

    all_res = np.concatenate([y_true - test_predictions[m] for m, _ in models_to_plot])
    max_abs_res = float(np.max(np.abs(all_res))) * 1.12

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for row_idx, (model_name, color) in enumerate(models_to_plot):
        y_pred = test_predictions[model_name]
        residuals = y_true - y_pred
        row_metrics = results_df.loc[results_df["model"] == model_name].iloc[0]

        # Left column: Predicted vs Actual
        ax_pred = axes[row_idx, 0]
        ax_pred.scatter(y_true, y_pred, alpha=0.65, s=26, color=color, edgecolors="white", linewidth=0.4)
        ax_pred.plot([min_val, max_val], [min_val, max_val], "k--", linewidth=1.2, label="Ideal (y = x)")
        ax_pred.set_xlim(min_val, max_val)
        ax_pred.set_ylim(min_val, max_val)
        ax_pred.set_xlabel("Actual Scaled SPL (dB)", fontsize=10)
        ax_pred.set_ylabel("Predicted Scaled SPL (dB)", fontsize=10)
        ax_pred.set_title(
            f"{model_name} — Predicted vs Actual (Hold-out Test)\n"
            f"RMSE = {row_metrics['test_rmse']:.2f} dB | MAE = {row_metrics['test_mae']:.2f} dB | R2 = {row_metrics['test_r2']:.3f}",
            fontsize=10,
        )
        ax_pred.grid(True, linestyle="--", alpha=0.35)
        ax_pred.legend(loc="upper left", frameon=True, fontsize=8.5)

        # Right column: Residuals vs Predicted
        ax_res = axes[row_idx, 1]
        ax_res.scatter(y_pred, residuals, alpha=0.65, s=26, color=color, edgecolors="white", linewidth=0.4)
        ax_res.axhline(0.0, color="black", linestyle="--", linewidth=1.2, label="Zero residual")
        ax_res.set_xlim(min_val, max_val)
        ax_res.set_ylim(-max_abs_res, max_abs_res)
        ax_res.set_xlabel("Predicted Scaled SPL (dB)", fontsize=10)
        ax_res.set_ylabel("Residual (Actual - Predicted, dB)", fontsize=10)
        ax_res.set_title(
            f"{model_name} — Residuals vs Predicted (Hold-out Test)\n"
            f"Mean = {residuals.mean():+.2f} dB | SD = {residuals.std(ddof=1):.2f} dB",
            fontsize=10,
        )
        ax_res.grid(True, linestyle="--", alpha=0.35)
        ax_res.legend(loc="upper right", frameon=True, fontsize=8.5)

    fig.suptitle(
        "Hold-out Test Residual Diagnostics: Best Linear Baseline vs Regularized Tree",
        fontsize=12.5,
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    out_path = figures_dir / "06_baseline_residuals.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def main() -> None:
    """Orchestrate preprocessing, GroupKFold baseline CV, leakage audit, and hold-out check."""
    set_reproducibility(RANDOM_SEED)
    _, data_path, figures_dir, tables_dir = resolve_paths()

    X, y, groups = load_dataset_and_groups(data_path)
    X_train, X_test, y_train, y_test, groups_train, _ = split_holdout_by_group(X, y, groups)

    _, prep_log_and_scale = build_preprocessors()
    pipelines = build_baseline_pipelines(prep_log_and_scale)

    # Phase 1: Model comparison strictly on X_train via 5-fold GroupKFold
    cv_df = evaluate_cv_baselines(pipelines, X_train, y_train, groups_train, n_splits=5)

    # Leakage-contrast audit on X_train
    run_leakage_contrast_check(pipelines, X_train, y_train, groups_train)

    # Phase 2: Single final Hold-out Test evaluation on untouched X_test, y_test
    test_df, test_predictions = evaluate_holdout_test_once(
        pipelines, X_train, y_train, X_test, y_test
    )

    results_df = cv_df.merge(test_df, on="model", how="left")
    ordered_cols = [
        "model",
        "cv_train_rmse",
        "cv_val_rmse",
        "cv_val_rmse_std",
        "cv_val_mae",
        "cv_val_r2",
        "overfit_gap_rmse",
        "test_rmse",
        "test_mae",
        "test_r2",
    ]
    results_df = results_df[ordered_cols]

    print("\n" + "=" * 78)
    print("3. BASELINE BENCHMARK SUMMARY (5-Fold GroupKFold CV + Final Hold-out Test)")
    print("=" * 78)
    print(results_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    out_csv = tables_dir / "baseline_results.csv"
    results_df.to_csv(out_csv, index=False, float_format="%.6f")
    print(f"\nExported baseline summary table -> {out_csv}")

    audit_test_residuals(
        y_test,
        test_predictions,
        ["Ridge_LogScaled", "DecisionTree_Regularized"],
    )

    fig5_path = plot_train_vs_val_rmse(results_df, figures_dir)
    fig6_path = plot_holdout_residuals(y_test, test_predictions, results_df, figures_dir)
    print(f"\nSaved publication-ready figures (300 DPI):")
    print(f"  - {fig5_path}")
    print(f"  - {fig6_path}")


if __name__ == "__main__":
    main()
