"""Exploratory Data Analysis and anti-leakage validation audit for NASA Airfoil Self-Noise.

Workflow stage: Understand -> Clean (audit only) -> Split (leakage audit).
No scalers, imputers, or models are fitted in this script.
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
import seaborn as sns
from scipy import stats
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import GroupKFold, KFold
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

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

UNITS: Dict[str, str] = {
    "frequency": "Hz",
    "angle_of_attack": "deg",
    "chord_length": "m",
    "free_stream_velocity": "m/s",
    "suction_side_displacement_thickness": "m",
    "scaled_sound_pressure_level": "dB",
    "log10_frequency": "log10(Hz)",
}


def set_reproducibility(seed: int = RANDOM_SEED) -> None:
    """Set deterministic random seeds for Python and NumPy."""
    random.seed(seed)
    np.random.seed(seed)


def resolve_paths() -> Tuple[Path, Path, Path, Path]:
    """Resolve project root, data file, figures directory, and tables directory."""
    project_root = Path(__file__).resolve().parents[1]
    data_path = project_root / "data" / "airfoil_self_noise.dat"
    figures_dir = project_root / "report" / "figures"
    tables_dir = project_root / "report" / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    return project_root, data_path, figures_dir, tables_dir


def load_and_audit_data(data_path: Path) -> pd.DataFrame:
    """Load raw dataset, verify schema/completeness, and run physical sanity checks."""
    df = pd.read_csv(data_path, sep=r"\s+", header=None, names=SCHEMA_COLUMNS)
    assert df.shape == (1503, 6), f"Expected shape (1503, 6), got {df.shape}"

    print("=" * 72)
    print("1. DATA LOADING & PHYSICAL PLAUSIBILITY AUDIT")
    print("=" * 72)
    print(f"Dataset shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print("\nColumn dtypes & missing values:")
    audit_df = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "missing": df.isna().sum(),
    })
    print(audit_df.to_string())
    print(f"\nExact duplicate rows: {int(df.duplicated().sum())}")

    sanity_rules = {
        "frequency > 0": df["frequency"] > 0,
        "chord_length > 0": df["chord_length"] > 0,
        "free_stream_velocity > 0": df["free_stream_velocity"] > 0,
        "suction_side_displacement_thickness > 0": (
            df["suction_side_displacement_thickness"] > 0
        ),
        "angle_of_attack >= 0": df["angle_of_attack"] >= 0,
    }
    print("\nPhysical domain sanity checks:")
    for rule_name, mask in sanity_rules.items():
        passed = int(mask.sum())
        failed = int((~mask).sum())
        print(f"  - {rule_name:<42}: pass={passed}, fail={failed}")

    return df


def compute_descriptive_stats(df: pd.DataFrame, tables_dir: Path) -> pd.DataFrame:
    """Compute descriptive statistics and export summary table to CSV."""
    records = []
    for col in SCHEMA_COLUMNS:
        series = df[col]
        q1 = float(series.quantile(0.25))
        q3 = float(series.quantile(0.75))
        records.append({
            "variable": col,
            "unit": UNITS[col],
            "mean": float(series.mean()),
            "std": float(series.std(ddof=1)),
            "median": float(series.median()),
            "iqr": q3 - q1,
            "min": float(series.min()),
            "max": float(series.max()),
            "skewness": float(series.skew()),
            "kurtosis": float(series.kurtosis()),
        })

    summary_df = pd.DataFrame(records)
    output_csv = tables_dir / "eda_summary.csv"
    summary_df.to_csv(output_csv, index=False, float_format="%.6f")

    print("\n" + "=" * 72)
    print("2. DESCRIPTIVE STATISTICS & NORMALITY DIAGNOSTICS")
    print("=" * 72)
    print(summary_df.to_string(index=False))
    print(f"\nExported summary table -> {output_csv}")
    return summary_df


def run_normality_and_skew_audit(
    df: pd.DataFrame, summary_df: pd.DataFrame, alpha: float = 0.05
) -> bool:
    """Run Shapiro-Wilk normality tests and flag skewed features for Pipeline transforms."""
    print(f"\nShapiro-Wilk normality tests (alpha = {alpha}):")
    any_non_normal = False
    for col in SCHEMA_COLUMNS:
        stat, p_val = stats.shapiro(df[col])
        is_normal = p_val >= alpha
        if not is_normal:
            any_non_normal = True
        status = "NORMAL" if is_normal else "NON-NORMAL (violation)"
        print(f"  - {col:<38}: W={stat:.4f}, p={p_val:.3e} -> {status}")

    if any_non_normal:
        print(
            "\n[FALLBACK TRIGGERED] Normality violated (p < 0.05) on >=1 variable. "
            "Using Spearman rank correlation as primary monotonic metric "
            "(Pearson reported for contrast only)."
        )

    skewed = summary_df[
        (summary_df["variable"].isin(FEATURE_COLUMNS))
        & (summary_df["skewness"].abs() > 1.0)
    ]
    print("\nFeatures with |skewness| > 1.0 (candidates for inside-Pipeline log/power transform):")
    if skewed.empty:
        print("  - None")
    else:
        for _, row in skewed.iterrows():
            print(f"  - {row['variable']:<38}: skewness={row['skewness']:.4f}")
    return any_non_normal


def compute_relevance_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """Compare Pearson, Spearman, and Mutual Information relevance against target."""
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    mi_scores = mutual_info_regression(X, y, random_state=RANDOM_SEED)
    rows = []
    for idx, col in enumerate(FEATURE_COLUMNS):
        pearson_r, _ = stats.pearsonr(df[col], y)
        spearman_rho, _ = stats.spearmanr(df[col], y)
        rows.append({
            "feature": col,
            "pearson_r": float(pearson_r),
            "spearman_rho": float(spearman_rho),
            "abs_spearman": float(abs(spearman_rho)),
            "mutual_info": float(mi_scores[idx]),
        })

    rel_df = (
        pd.DataFrame(rows)
        .sort_values(by="mutual_info", ascending=False)
        .reset_index(drop=True)
    )
    print("\nFeature relevance comparison vs scaled_sound_pressure_level (ranked by MI):")
    print(rel_df.to_string(index=False))
    print(
        "[NOTE] Mutual Information is used here strictly as an exploratory filter diagnostic; "
        "any actual feature selection must occur inside CV folds during modeling."
    )
    return rel_df


def compute_vif_diagnostics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Variance Inflation Factors on the 5 features (with intercept)."""
    X_const = add_constant(df[FEATURE_COLUMNS], has_constant="add")
    vif_rows = []
    for i, col in enumerate(X_const.columns):
        if col == "const":
            continue
        vif_val = float(variance_inflation_factor(X_const.values, i))
        vif_rows.append({"feature": col, "VIF": vif_val})

    vif_df = pd.DataFrame(vif_rows).sort_values(by="VIF", ascending=False).reset_index(drop=True)
    print("\nMulticollinearity audit (VIF on 5 features with intercept):")
    print(vif_df.to_string(index=False))
    return vif_df


def audit_outliers_tails_and_balance(df: pd.DataFrame) -> None:
    """Quantify 1.5*IQR outliers, target deciles, and discrete experimental level counts."""
    print("\nStatistical outlier audit (1.5 * IQR rule; NO rows dropped):")
    for col in SCHEMA_COLUMNS:
        q1 = float(df[col].quantile(0.25))
        q3 = float(df[col].quantile(0.75))
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        mask = (df[col] < lower) | (df[col] > upper)
        n_out = int(mask.sum())
        pct_out = 100.0 * n_out / len(df)
        print(
            f"  - {col:<38}: {n_out:3d} ({pct_out:5.2f}%) "
            f"[bounds: {lower:.4g}, {upper:.4g}; actual: {df[col].min():.4g}..{df[col].max():.4g}]"
        )

    deciles = np.quantile(df[TARGET_COLUMN], np.linspace(0.0, 1.0, 11))
    print("\nTarget tail coverage (scaled_sound_pressure_level deciles in dB):")
    decile_str = ", ".join(f"D{i*10}={val:.2f}" for i, val in enumerate(deciles))
    print(f"  {decile_str}")

    print("\nSample balance across discrete experimental levels:")
    for col in ["chord_length", "free_stream_velocity"]:
        counts = df[col].value_counts().sort_index()
        formatted = ", ".join(f"{k:g} {UNITS[col]}: {int(v)}" for k, v in counts.items())
        print(f"  - {col}: {formatted}")


def audit_experimental_groups_and_leakage(df: pd.DataFrame) -> pd.Series:
    """Define aerodynamic configuration groups and compare KFold vs GroupKFold leakage."""
    print("\n" + "=" * 72)
    print("3. EXPERIMENTAL GROUPING & DATA LEAKAGE AUDIT")
    print("=" * 72)

    group_ids = df.groupby(GROUP_COLUMNS, sort=False).ngroup()
    sweep_lengths = group_ids.value_counts()
    print(f"Total unique aerodynamic configurations: {group_ids.nunique()}")
    print(
        f"Sweep lengths (samples per configuration): "
        f"min={int(sweep_lengths.min())}, "
        f"median={float(sweep_lengths.median()):.1f}, "
        f"max={int(sweep_lengths.max())}"
    )

    kfold = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    gkfold = GroupKFold(n_splits=5)

    print("\n5-Fold Leakage Audit (% of validation rows whose config appears in train fold):")
    for cv_name, splitter, groups in [
        ("KFold(shuffle=True, seed=42)", kfold, None),
        ("GroupKFold(n_splits=5)", gkfold, group_ids),
    ]:
        fold_pcts = []
        print(f"  {cv_name}:")
        for fold_idx, (train_idx, val_idx) in enumerate(
            splitter.split(df[FEATURE_COLUMNS], df[TARGET_COLUMN], groups=groups), start=1
        ):
            train_groups = set(group_ids.iloc[train_idx])
            val_groups = group_ids.iloc[val_idx]
            leaked = int(val_groups.isin(train_groups).sum())
            pct = 100.0 * leaked / len(val_idx)
            fold_pcts.append(pct)
            print(f"    Fold {fold_idx}: {leaked}/{len(val_idx)} ({pct:6.2f}%)")
        print(f"    Mean leakage: {np.mean(fold_pcts):6.2f}%")

    return group_ids


def plot_distributions(df: pd.DataFrame, figures_dir: Path) -> None:
    """Save 01_distributions.png: histograms + KDE for all 6 variables + log10(frequency)."""
    plot_cols = SCHEMA_COLUMNS + ["log10_frequency"]
    plot_df = df.copy()
    plot_df["log10_frequency"] = np.log10(plot_df["frequency"])

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes_flat = axes.ravel()

    for idx, col in enumerate(plot_cols):
        ax = axes_flat[idx]
        sns.histplot(plot_df[col], kde=True, bins=25, ax=ax, color="#2b5c8f", edgecolor="white")
        ax.set_title(col.replace("_", " ").title(), fontsize=10)
        ax.set_xlabel(f"{col} ({UNITS[col]})", fontsize=9)
        ax.set_ylabel("Count", fontsize=9)

    axes_flat[-1].axis("off")
    fig.suptitle("NASA Airfoil Self-Noise: Variable Distributions (n = 1,503)", fontsize=13, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = figures_dir / "01_distributions.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_spearman_heatmap(df: pd.DataFrame, figures_dir: Path) -> None:
    """Save 02_spearman_correlation.png: annotated lower-triangle Spearman heatmap."""
    corr = df[SCHEMA_COLUMNS].corr(method="spearman")
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1.0,
        vmax=1.0,
        center=0.0,
        square=True,
        linewidths=0.5,
        cbar_kws={"label": "Spearman rho"},
        ax=ax,
    )
    ax.set_title("Spearman Rank Correlation Matrix (Lower Triangle)", fontsize=12, pad=12)
    fig.tight_layout()
    out_path = figures_dir / "02_spearman_correlation.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_feature_relevance(rel_df: pd.DataFrame, figures_dir: Path) -> None:
    """Save 03_feature_relevance.png: |Spearman rho| vs Mutual Information horizontal bars."""
    ordered = rel_df.sort_values(by="mutual_info", ascending=True)
    y_pos = np.arange(len(ordered))
    bar_height = 0.36

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(
        y_pos + bar_height / 2,
        ordered["abs_spearman"],
        height=bar_height,
        label="|Spearman rho| (monotonic)",
        color="#3b75af",
    )
    ax.barh(
        y_pos - bar_height / 2,
        ordered["mutual_info"],
        height=bar_height,
        label="Mutual Information (non-linear)",
        color="#e07a38",
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(ordered["feature"], fontsize=9)
    ax.set_xlabel("Relevance Score (dimensionless / nats)", fontsize=10)
    ax.set_title("Feature Relevance vs Scaled Sound Pressure Level (dB)", fontsize=12)
    ax.legend(loc="lower right", frameon=True)
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    fig.tight_layout()
    out_path = figures_dir / "03_feature_relevance.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_acoustic_spectra(df: pd.DataFrame, figures_dir: Path) -> None:
    """Save 04_acoustic_spectra.png: SPL vs log frequency colored by AoA at c=0.3048m, U=71.3m/s."""
    subset = df[
        np.isclose(df["chord_length"], 0.3048)
        & np.isclose(df["free_stream_velocity"], 71.3)
    ].sort_values(by=["angle_of_attack", "frequency"])

    fig, ax = plt.subplots(figsize=(9, 5.5))
    angles = sorted(subset["angle_of_attack"].unique())
    palette = sns.color_palette("viridis", n_colors=len(angles))

    for color, aoa in zip(palette, angles):
        aoa_df = subset[np.isclose(subset["angle_of_attack"], aoa)]
        ax.plot(
            aoa_df["frequency"],
            aoa_df["scaled_sound_pressure_level"],
            marker="o",
            markersize=4,
            linewidth=1.5,
            color=color,
            label=f"AoA = {aoa:g} deg",
        )

    ax.set_xscale("log")
    ax.set_xlabel("Frequency (Hz, log scale)", fontsize=10)
    ax.set_ylabel("Scaled Sound Pressure Level (dB)", fontsize=10)
    ax.set_title(
        "Acoustic Spectra Across Angles of Attack (c = 0.3048 m, U = 71.3 m/s)",
        fontsize=11,
    )
    ax.legend(title="Angle of Attack", fontsize=8, title_fontsize=9, ncol=2, frameon=True)
    ax.grid(True, which="both", linestyle="--", alpha=0.35)
    fig.tight_layout()
    out_path = figures_dir / "04_acoustic_spectra.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main() -> None:
    """Orchestrate the full EDA and anti-leakage validation audit."""
    set_reproducibility(RANDOM_SEED)
    _, data_path, figures_dir, tables_dir = resolve_paths()

    df = load_and_audit_data(data_path)
    summary_df = compute_descriptive_stats(df, tables_dir)
    run_normality_and_skew_audit(df, summary_df)
    rel_df = compute_relevance_comparison(df)
    compute_vif_diagnostics(df)
    audit_outliers_tails_and_balance(df)
    audit_experimental_groups_and_leakage(df)

    plot_distributions(df, figures_dir)
    plot_spearman_heatmap(df, figures_dir)
    plot_feature_relevance(rel_df, figures_dir)
    plot_acoustic_spectra(df, figures_dir)
    print(f"\nSaved 4 publication-ready figures (300 DPI) -> {figures_dir}")


if __name__ == "__main__":
    main()
