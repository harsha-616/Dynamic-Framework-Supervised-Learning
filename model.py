"""
=============================================================
  Dynamic Framework: Supervised Learning
  ANN trained on 3 datasets × multiple activation functions
=============================================================

Datasets:
  1. Iris              – Multiclass Classification  (150 samples, 4 features)
  2. Breast Cancer     – Binary Classification      (569 samples, 30 features)
  3. California Housing– Regression                 (20640 samples, 8 features)

Activation Functions: relu, tanh, logistic (sigmoid), identity

HOW TO RUN:
  pip install -r requirements.txt
  python dynamic_framework.py

OUTPUTS:
  ann_comparison.png   – full visual comparison chart
  results_report.csv   – all metrics in a table
=============================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
import time
import os

from sklearn.datasets import load_iris, load_breast_cancer, fetch_california_housing
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score
)

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# 1. CONFIGURATION  ← Edit here to change things dynamically
# ─────────────────────────────────────────────────────────────

ACTIVATION_FUNCTIONS = ["relu", "tanh", "logistic", "identity"]

HIDDEN_LAYERS  = (128, 64, 32)   # ANN architecture (shared for all)
MAX_ITER       = 500
RANDOM_STATE   = 42
TEST_SIZE      = 0.2
OUTPUT_DIR     = "."             # folder to save outputs (current directory)

# ─────────────────────────────────────────────────────────────
# 2. DATASET LOADER
#    → Add more datasets here to extend the framework!
# ─────────────────────────────────────────────────────────────

def load_datasets():
    """
    Returns a list of dataset config dicts.
    Each dict must have: name, task, X, y, feature_names, description.
    task must be 'classification' or 'regression'.
    """
    datasets = []

    # ── Dataset 1: Iris (Multiclass Classification) ──
    iris = load_iris()
    datasets.append({
        "name":          "Iris",
        "source":        "sklearn / Kaggle",
        "task":          "classification",
        "n_classes":     3,
        "X":             iris.data,
        "y":             iris.target,
        "feature_names": list(iris.feature_names),
        "target_names":  list(iris.target_names),
        "description":   "4 features, 150 samples, 3 classes (flower species)"
    })

    # ── Dataset 2: Breast Cancer Wisconsin (Binary Classification) ──
    bc = load_breast_cancer()
    datasets.append({
        "name":          "Breast Cancer",
        "source":        "sklearn / Kaggle",
        "task":          "classification",
        "n_classes":     2,
        "X":             bc.data,
        "y":             bc.target,
        "feature_names": list(bc.feature_names),
        "target_names":  list(bc.target_names),
        "description":   "30 features, 569 samples, 2 classes (malignant/benign)"
    })

    # ── Dataset 3: California Housing (Regression) ──
    housing = fetch_california_housing()
    datasets.append({
        "name":          "California Housing",
        "source":        "sklearn / Kaggle",
        "task":          "regression",
        "X":             housing.data,
        "y":             housing.target,
        "feature_names": list(housing.feature_names),
        "target_names":  ["MedHouseVal"],
        "description":   "8 features, 20640 samples — predict median house value"
    })

    return datasets

# ─────────────────────────────────────────────────────────────
# 3. MODEL BUILDER  (swaps activation function dynamically)
# ─────────────────────────────────────────────────────────────

def build_model(task: str, activation: str):
    """Build an MLP model for the given task and activation function."""
    params = dict(
        hidden_layer_sizes  = HIDDEN_LAYERS,
        activation          = activation,
        max_iter            = MAX_ITER,
        random_state        = RANDOM_STATE,
        early_stopping      = True,
        validation_fraction = 0.1,
        n_iter_no_change    = 20,
    )
    if task == "classification":
        return MLPClassifier(**params)
    else:
        return MLPRegressor(**params)

# ─────────────────────────────────────────────────────────────
# 4. EVALUATOR
# ─────────────────────────────────────────────────────────────

def evaluate(model, X_test, y_test, task):
    """Return all relevant metrics for a trained model."""
    y_pred  = model.predict(X_test)
    metrics = {}

    if task == "classification":
        metrics["Accuracy"]   = accuracy_score(y_test, y_pred)
        metrics["F1 (macro)"] = f1_score(y_test, y_pred, average="macro")
        metrics["Iterations"] = model.n_iter_
        metrics["Best Loss"]  = model.best_loss_
    else:
        mse = mean_squared_error(y_test, y_pred)
        metrics["MSE"]        = mse
        metrics["RMSE"]       = np.sqrt(mse)
        metrics["MAE"]        = mean_absolute_error(y_test, y_pred)
        metrics["R2"]         = r2_score(y_test, y_pred)
        metrics["Iterations"] = model.n_iter_
        metrics["Best Loss"]  = model.best_loss_

    return metrics, y_pred

# ─────────────────────────────────────────────────────────────
# 5. MAIN EXPERIMENT LOOP  (The "Dynamic" core)
# ─────────────────────────────────────────────────────────────

def run_experiments():
    datasets    = load_datasets()
    all_results = []

    print("\n" + "="*70)
    print("   DYNAMIC FRAMEWORK: SUPERVISED LEARNING — ANN EXPERIMENTS")
    print("="*70)
    print(f"  Architecture : {HIDDEN_LAYERS}")
    print(f"  Activations  : {ACTIVATION_FUNCTIONS}")
    print(f"  Max Epochs   : {MAX_ITER}")
    print(f"  Test Split   : {int(TEST_SIZE*100)}%")
    print("="*70)

    for ds in datasets:
        print(f"\n\n{'─'*60}")
        print(f"  Dataset : {ds['name']}")
        print(f"  Task    : {ds['task'].upper()}")
        print(f"  Info    : {ds['description']}")
        print(f"{'─'*60}")

        X, y = ds["X"], ds["y"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size    = TEST_SIZE,
            random_state = RANDOM_STATE,
            stratify     = y if ds["task"] == "classification" else None
        )

        scaler  = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)

        ds_results = {
            "dataset": ds["name"],
            "task":    ds["task"],
            "models":  {}
        }

        for act in ACTIVATION_FUNCTIONS:
            print(f"\n  ▶  Activation : {act.upper():<12}", end="", flush=True)
            model = build_model(ds["task"], act)

            t0      = time.time()
            model.fit(X_train, y_train)
            elapsed = time.time() - t0

            metrics, y_pred = evaluate(model, X_test, y_test, ds["task"])
            metrics["Train Time (s)"] = round(elapsed, 3)

            ds_results["models"][act] = {
                "metrics":    metrics,
                "y_pred":     y_pred,
                "y_test":     y_test,
                "loss_curve": model.loss_curve_,
                "model":      model,
            }

            if ds["task"] == "classification":
                print(f"  Acc={metrics['Accuracy']:.4f}  "
                      f"F1={metrics['F1 (macro)']:.4f}  "
                      f"Iter={metrics['Iterations']}  "
                      f"Time={elapsed:.2f}s")
            else:
                print(f"  R2={metrics['R2']:.4f}  "
                      f"RMSE={metrics['RMSE']:.4f}  "
                      f"Iter={metrics['Iterations']}  "
                      f"Time={elapsed:.2f}s")

        all_results.append(ds_results)

    return all_results, datasets

# ─────────────────────────────────────────────────────────────
# 6. RESULTS DATAFRAME
# ─────────────────────────────────────────────────────────────

def build_results_dataframe(all_results):
    rows = []
    for ds_res in all_results:
        for act, data in ds_res["models"].items():
            row = {
                "Dataset":    ds_res["dataset"],
                "Task":       ds_res["task"],
                "Activation": act,
                **{k: v for k, v in data["metrics"].items()}
            }
            rows.append(row)
    return pd.DataFrame(rows)

# ─────────────────────────────────────────────────────────────
# 7. VISUALIZATION
# ─────────────────────────────────────────────────────────────

PALETTE = {
    "relu":     "#4FC3F7",
    "tanh":     "#FF8A65",
    "logistic": "#A5D6A7",
    "identity": "#CE93D8",
}

def plot_all(all_results, datasets, df):
    n_ds = len(all_results)

    fig = plt.figure(figsize=(22, 28), facecolor="#0F0F1A")
    fig.suptitle(
        "Dynamic Framework: Supervised Learning\nANN Activation Function Comparison",
        fontsize=22, fontweight="bold", color="white", y=0.985
    )

    outer = gridspec.GridSpec(
        4, n_ds, figure=fig,
        hspace=0.55, wspace=0.35,
        top=0.94, bottom=0.04,
        left=0.06, right=0.97
    )

    TEXT = "#E0E0E0"
    GRID = "#2A2A3E"
    BG   = "#16162A"

    def style_ax(ax, title="", xlabel="", ylabel=""):
        ax.set_facecolor(BG)
        ax.tick_params(colors=TEXT, labelsize=8)
        ax.xaxis.label.set_color(TEXT)
        ax.yaxis.label.set_color(TEXT)
        ax.title.set_color(TEXT)
        for spine in ax.spines.values():
            spine.set_color(GRID)
        ax.grid(color=GRID, linewidth=0.5, alpha=0.7)
        if title:  ax.set_title(title,  fontsize=10, pad=6, color=TEXT)
        if xlabel: ax.set_xlabel(xlabel, fontsize=8)
        if ylabel: ax.set_ylabel(ylabel, fontsize=8)

    # ── Row 0: Performance bar charts ──
    for col, ds_res in enumerate(all_results):
        ax   = fig.add_subplot(outer[0, col])
        task = ds_res["task"]
        acts = list(ds_res["models"].keys())

        if task == "classification":
            vals   = [ds_res["models"][a]["metrics"]["Accuracy"] for a in acts]
            metric = "Accuracy"
        else:
            vals   = [ds_res["models"][a]["metrics"]["R2"] for a in acts]
            metric = "R2 Score"

        ylim   = (max(0, min(vals) - 0.05), 1.01)
        colors = [PALETTE[a] for a in acts]
        bars   = ax.bar(acts, vals, color=colors, width=0.6, zorder=3)

        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom",
                    fontsize=8, color=TEXT, fontweight="bold")

        ax.set_ylim(ylim)
        style_ax(ax, title=f"{ds_res['dataset']}\n{metric}",
                 xlabel="Activation", ylabel=metric)
        ax.set_xticklabels(acts, rotation=20, ha="right")

    # ── Row 1: Loss curves ──
    for col, ds_res in enumerate(all_results):
        ax = fig.add_subplot(outer[1, col])
        for act, data in ds_res["models"].items():
            ax.plot(data["loss_curve"], label=act,
                    color=PALETTE[act], linewidth=1.8, alpha=0.9)
        style_ax(ax, title=f"{ds_res['dataset']}\nTraining Loss Curve",
                 xlabel="Epoch", ylabel="Loss")
        ax.legend(fontsize=7, labelcolor=TEXT, facecolor=GRID, edgecolor=GRID)

    # ── Row 2: Metrics heatmap ──
    for col, (ds_res, ds) in enumerate(zip(all_results, datasets)):
        ax   = fig.add_subplot(outer[2, col])
        task = ds_res["task"]
        acts = list(ds_res["models"].keys())

        metric_keys = (["Accuracy", "F1 (macro)", "Best Loss", "Iterations"]
                       if task == "classification"
                       else ["R2", "RMSE", "MAE", "Iterations"])

        heat_data = [
            [ds_res["models"][a]["metrics"].get(m, np.nan) for m in metric_keys]
            for a in acts
        ]
        heat_df = pd.DataFrame(heat_data, index=acts, columns=metric_keys)

        normed = heat_df.copy()
        for c in heat_df.columns:
            lo, hi = heat_df[c].min(), heat_df[c].max()
            normed[c] = (heat_df[c] - lo) / (hi - lo) if hi != lo else 0.5

        annot_df = heat_df.map(
            lambda x: f"{x:.3f}" if isinstance(x, (int, float)) and not pd.isna(x) else "N/A"
        )
        sns.heatmap(normed.astype(float), ax=ax, cmap="YlOrRd",
                    annot=annot_df, fmt="", linewidths=0.5,
                    cbar=False, annot_kws={"size": 7, "color": "#1a1a1a"})
        ax.set_title(f"{ds_res['dataset']}\nMetrics Heatmap",
                     fontsize=10, color=TEXT, pad=6)
        ax.tick_params(colors=TEXT, labelsize=8)
        ax.set_facecolor(BG)

    # ── Row 3: Training time + iterations ──
    for col, ds_res in enumerate(all_results):
        ax    = fig.add_subplot(outer[3, col])
        acts  = list(ds_res["models"].keys())
        times = [ds_res["models"][a]["metrics"]["Train Time (s)"] for a in acts]
        iters = [ds_res["models"][a]["metrics"]["Iterations"]     for a in acts]

        ax2 = ax.twinx()
        ax2.set_facecolor(BG)

        ax.bar(acts, times, color=[PALETTE[a] for a in acts],
               width=0.5, zorder=3, alpha=0.85)
        ax2.plot(acts, iters, color="white", marker="o",
                 linewidth=1.5, markersize=5, zorder=4, label="Iterations")

        style_ax(ax, title=f"{ds_res['dataset']}\nTrain Time & Iterations",
                 xlabel="Activation", ylabel="Time (s)")
        ax2.tick_params(colors=TEXT, labelsize=8)
        ax2.set_ylabel("Iterations", fontsize=8, color=TEXT)
        for spine in ax2.spines.values():
            spine.set_color(GRID)
        ax.set_xticklabels(acts, rotation=20, ha="right")
        ax2.legend(fontsize=7, labelcolor=TEXT, facecolor=GRID, edgecolor=GRID)

    out_path = os.path.join(OUTPUT_DIR, "ann_comparison.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"\n  ✔  Plot saved   → {out_path}")
    plt.close()

# ─────────────────────────────────────────────────────────────
# 8. SAVE CSV + PRINT SUMMARY
# ─────────────────────────────────────────────────────────────

def save_report(df):
    csv_path = os.path.join(OUTPUT_DIR, "results_report.csv")
    df.to_csv(csv_path, index=False)
    print(f"  ✔  Report saved → {csv_path}")

    print("\n" + "="*70)
    print("  FINAL RESULTS TABLE")
    print("="*70)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 130)
    pd.set_option("display.float_format", "{:.4f}".format)
    print(df.to_string(index=False))

    print("\n" + "="*70)
    print("  BEST ACTIVATION FUNCTION PER DATASET")
    print("="*70)
    for ds_name in df["Dataset"].unique():
        sub  = df[df["Dataset"] == ds_name]
        task = sub["Task"].iloc[0]
        if task == "classification":
            best = sub.loc[sub["Accuracy"].idxmax()]
            print(f"  {ds_name:<24} → {best['Activation']:<12} "
                  f"Accuracy={best['Accuracy']:.4f}  F1={best['F1 (macro)']:.4f}")
        else:
            best = sub.loc[sub["R2"].idxmax()]
            print(f"  {ds_name:<24} → {best['Activation']:<12} "
                  f"R2={best['R2']:.4f}  RMSE={best['RMSE']:.4f}")

# ─────────────────────────────────────────────────────────────
# 9. ENTRY POINT
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    all_results, datasets = run_experiments()
    df = build_results_dataframe(all_results)

    print("\n\n  Generating visualizations...")
    plot_all(all_results, datasets, df)
    save_report(df)

    print("\n  ✔  All done!\n")
