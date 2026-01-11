# scripts/make_plots.py
# Generate Results & Discussion plots from outputs/runs/*.json
#
# Run:
#   python scripts/make_plots.py
#
# Output:
#   outputs/figures/*.png
#   outputs/figures/summary_metrics.csv
#
# What this script can plot (auto-skips if data not present):
# 1) Supported vs Under-specified vs Contradicted (bar + stacked bar)
# 2) Distribution of SLM p_yes (overall + by predicted label)
# 3) Retrieval similarity distribution (sim_max)
# 4) Scatter: sim_max vs p_yes (colored by predicted label)
# 5) If gold labels exist:
#    - Confusion matrix (heatmap)
#    - Precision/Recall/F1 bar chart
#    - ROC & PR curves (+ AUC/AP)
#    - Threshold sensitivity (threshold vs F1)
#    - Calibration curve + ECE
# 6) If latency data exists:
#    - Latency per layer (stacked bar)
# 7) If ablation/variant data exists:
#    - Ablation comparison (F1 per variant)
# 8) If rule flags exist:
#    - Error type pie chart
#
# Expected optional fields in run JSON (recommended for full paper graphs):
# - run["timings"] = {"l0_llm":..., "l1_retrieval":..., "l2_slm":..., "l3_reasoner":..., "total":...}
# - run["variant"] = "full" | "no_l1" | "no_l2" | "no_l3" | "baseline" etc.
# - per_fact[i]["gold_label"] in {"SUPPORTED","CONTRADICTED","UNDER_SPECIFIED"}  (ground truth)
# - per_fact[i]["rule_flag"] (for error analysis), e.g. "MISSING_UNIT"
#
# NOTE:
# - Your current pipeline logs predicted labels in per_fact[i]["gate_label"]
# - And scores like p_yes, p_no, sim_max, uncertainty

import os
import glob
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

# Seaborn makes nicer plots; fall back to matplotlib-only if not installed
try:
    import seaborn as sns
    _HAS_SEABORN = True
except Exception:
    sns = None
    _HAS_SEABORN = False

from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
    roc_curve,
    precision_recall_curve,
    auc,
    average_precision_score,
)

RUNS_GLOB = "outputs/runs/run_*.json"
OUT_DIR = Path("outputs/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ----------------------------
# Utils
# ----------------------------

def _read_json(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _to_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return np.nan


def _safe_get(d: Dict[str, Any], key: str, default=None):
    return d.get(key, default) if isinstance(d, dict) else default


def _save_fig(fig: plt.Figure, name: str):
    out = OUT_DIR / name
    fig.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)
    print(f"[OK] {out}")


def _label_order() -> List[str]:
    return ["SUPPORTED", "UNDER_SPECIFIED", "CONTRADICTED"]


def load_runs() -> List[Dict[str, Any]]:
    paths = sorted(glob.glob(RUNS_GLOB))
    runs = []
    for p in paths:
        data = _read_json(p)
        if not data:
            continue
        data["_run_path"] = p
        runs.append(data)
    if not runs:
        print(f"[WARN] No run files found: {RUNS_GLOB}")
    return runs


def flatten_runs(runs: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for r in runs:
        run_path = r.get("_run_path", "")
        query = _safe_get(r, "query", "")
        variant = _safe_get(r, "variant", None)
        domain = _safe_get(r, "domain", None)

        timings = _safe_get(r, "timings", {}) or {}

        per_fact = _safe_get(r, "per_fact", []) or []
        for i, f in enumerate(per_fact):
            gate_label = _safe_get(f, "gate_label", _safe_get(f, "slm_status", "UNKNOWN"))
            gold_label = _safe_get(f, "gold_label", None)

            p_yes = _safe_get(f, "p_yes", _safe_get(f, "slm_yes_prob", np.nan))
            p_no = _safe_get(f, "p_no", np.nan)
            sim_max = _safe_get(f, "sim_max", np.nan)
            uncertainty = _safe_get(f, "uncertainty", np.nan)

            # Infer sim_max if missing
            if (sim_max is None or (isinstance(sim_max, float) and np.isnan(sim_max))) and isinstance(f.get("retrieved"), list):
                scores = []
                for it in f["retrieved"]:
                    if isinstance(it, dict) and "score" in it:
                        scores.append(_to_float(it["score"]))
                scores = [s for s in scores if not np.isnan(s)]
                sim_max = max(scores) if scores else np.nan

            # Optional error/rule tags
            rule_flag = _safe_get(f, "rule_flag", None)
            if rule_flag is None:
                # if you stored it in reasoning JSON or rule dict, you can map later
                rule_flag = _safe_get(_safe_get(f, "rule", {}), "rule_flag", None)

            rows.append({
                "run_path": run_path,
                "query": query,
                "variant": variant,
                "domain": domain,

                "fact_index": i,
                "fact": _safe_get(f, "fact", ""),

                "pred_label": str(gate_label) if gate_label is not None else "UNKNOWN",
                "gold_label": str(gold_label) if gold_label is not None else None,

                "p_yes": _to_float(p_yes),
                "p_no": _to_float(p_no),
                "sim_max": _to_float(sim_max),
                "uncertainty": _to_float(uncertainty),

                "rule_flag": rule_flag,

                # timings (repeated per fact for convenience)
                "t_l0_llm": _to_float(timings.get("l0_llm", np.nan)),
                "t_l1_retrieval": _to_float(timings.get("l1_retrieval", np.nan)),
                "t_l2_slm": _to_float(timings.get("l2_slm", np.nan)),
                "t_l3_reasoner": _to_float(timings.get("l3_reasoner", np.nan)),
                "t_total": _to_float(timings.get("total", np.nan)),
            })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Normalize label strings
    df["pred_label"] = df["pred_label"].fillna("UNKNOWN")
    return df


# ----------------------------
# Plot helpers
# ----------------------------

def plot_pred_label_bar(df: pd.DataFrame):
    counts = df["pred_label"].value_counts()
    # Ensure consistent order
    order = [l for l in _label_order() if l in counts.index] + [l for l in counts.index if l not in _label_order()]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.bar(order, [counts.get(l, 0) for l in order])
    ax.set_title("Predicted Labels (Gate Output)")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Count")
    _save_fig(fig, "01_pred_label_bar.png")


def plot_pred_label_stacked_by_query(df: pd.DataFrame, top_n: int = 20):
    # Group by query: counts of labels, pick top_n queries by total facts
    g = df.groupby(["query", "pred_label"]).size().unstack(fill_value=0)
    g["__total__"] = g.sum(axis=1)
    g = g.sort_values("__total__", ascending=False).head(top_n).drop(columns="__total__")

    # reorder columns
    cols = [c for c in _label_order() if c in g.columns] + [c for c in g.columns if c not in _label_order()]
    g = g[cols]

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)
    bottom = np.zeros(len(g))
    x = np.arange(len(g))

    for c in g.columns:
        ax.bar(x, g[c].values, bottom=bottom, label=c)
        bottom += g[c].values

    ax.set_title(f"Predicted Label Mix per Query (Top {top_n} queries)")
    ax.set_xlabel("Query index (top queries)")
    ax.set_ylabel("# facts")
    ax.legend()
    ax.set_xticks([])
    _save_fig(fig, "02_pred_label_stacked_by_query.png")


def plot_hist(df: pd.DataFrame, col: str, title: str, filename: str, by_label: bool = False):
    d = df.dropna(subset=[col])
    if d.empty:
        return

    if _HAS_SEABORN:
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        if by_label and "pred_label" in d.columns:
            sns.histplot(data=d, x=col, hue="pred_label", kde=True, ax=ax)
        else:
            sns.histplot(data=d, x=col, kde=True, ax=ax)
        ax.set_title(title)
        _save_fig(fig, filename)
    else:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.hist(d[col].values, bins=30)
        ax.set_title(title)
        ax.set_xlabel(col)
        ax.set_ylabel("Frequency")
        _save_fig(fig, filename)


def plot_scatter(df: pd.DataFrame, x: str, y: str, title: str, filename: str):
    d = df.dropna(subset=[x, y])
    if d.empty:
        return

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)

    if _HAS_SEABORN:
        sns.scatterplot(data=d, x=x, y=y, hue="pred_label", ax=ax)
    else:
        for lab, sub in d.groupby("pred_label"):
            ax.scatter(sub[x].values, sub[y].values, label=str(lab), alpha=0.7)
        ax.legend()

    ax.set_title(title)
    _save_fig(fig, filename)


# ----------------------------
# Supervised metrics (needs gold_label)
# ----------------------------

def has_gold(df: pd.DataFrame) -> bool:
    return "gold_label" in df.columns and df["gold_label"].notna().any()


def plot_confusion_matrix(df: pd.DataFrame):
    if not has_gold(df):
        return
    d = df.dropna(subset=["gold_label"])
    labels = _label_order()
    y_true = d["gold_label"].astype(str)
    y_pred = d["pred_label"].astype(str)

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111)

    if _HAS_SEABORN:
        sns.heatmap(cm, annot=True, fmt="d", cbar=True, xticklabels=labels, yticklabels=labels, ax=ax)
    else:
        im = ax.imshow(cm)
        fig.colorbar(im, ax=ax)
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_yticklabels(labels)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    ax.set_title("Confusion Matrix (Gold vs Predicted)")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Gold")
    _save_fig(fig, "20_confusion_matrix.png")


def classification_report_metrics(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    if not has_gold(df):
        return None
    d = df.dropna(subset=["gold_label"])
    labels = _label_order()

    y_true = d["gold_label"].astype(str)
    y_pred = d["pred_label"].astype(str)

    p, r, f1, s = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    rep = pd.DataFrame({
        "label": labels,
        "precision": p,
        "recall": r,
        "f1": f1,
        "support": s,
    })
    # micro/macro
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    p_micro, r_micro, f1_micro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="micro", zero_division=0
    )
    rep2 = pd.DataFrame([
        {"label": "macro_avg", "precision": p_macro, "recall": r_macro, "f1": f1_macro, "support": int(s.sum())},
        {"label": "micro_avg", "precision": p_micro, "recall": r_micro, "f1": f1_micro, "support": int(s.sum())},
    ])
    return pd.concat([rep, rep2], ignore_index=True)


def plot_pr_roc(df: pd.DataFrame):
    """
    Binary view: SUPPORTED as positive class.
    Uses p_yes as score.
    """
    if not has_gold(df):
        return
    d = df.dropna(subset=["gold_label", "p_yes"])
    if d.empty:
        return

    y_true = (d["gold_label"].astype(str) == "SUPPORTED").astype(int).values
    y_score = d["p_yes"].astype(float).values

    # Need both classes
    if len(set(y_true)) < 2:
        return

    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111)
    ax.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_title("ROC Curve (SUPPORTED vs Others)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    _save_fig(fig, "21_roc_curve.png")

    # PR
    prec, rec, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111)
    ax.plot(rec, prec, label=f"AP={ap:.3f}")
    ax.set_title("Precision–Recall Curve (SUPPORTED vs Others)")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend()
    _save_fig(fig, "22_pr_curve.png")


def plot_threshold_vs_f1(df: pd.DataFrame):
    """
    Threshold sweep: convert score -> predicted supported if p_yes >= thr
    Then compute F1 vs threshold (binary; supported positive).
    """
    if not has_gold(df):
        return
    d = df.dropna(subset=["gold_label", "p_yes"])
    if d.empty:
        return

    y_true = (d["gold_label"].astype(str) == "SUPPORTED").astype(int).values
    scores = d["p_yes"].astype(float).values

    if len(set(y_true)) < 2:
        return

    thresholds = np.linspace(0.0, 1.0, 101)
    f1s = []
    for t in thresholds:
        y_pred = (scores >= t).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
        f1s.append(f1)

    fig = plt.figure(figsize=(9, 5))
    ax = fig.add_subplot(111)
    ax.plot(thresholds, f1s)
    ax.set_title("Threshold Sensitivity: F1 vs p_yes Threshold")
    ax.set_xlabel("p_yes threshold")
    ax.set_ylabel("F1")
    _save_fig(fig, "23_threshold_vs_f1.png")


def calibration_curve_and_ece(df: pd.DataFrame, n_bins: int = 10):
    """
    Reliability diagram + Expected Calibration Error (ECE)
    (Binary: supported vs not)
    """
    if not has_gold(df):
        return
    d = df.dropna(subset=["gold_label", "p_yes"])
    if d.empty:
        return

    y_true = (d["gold_label"].astype(str) == "SUPPORTED").astype(int).values
    p = np.clip(d["p_yes"].astype(float).values, 0, 1)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(p, bins) - 1
    bin_ids = np.clip(bin_ids, 0, n_bins - 1)

    bin_conf = []
    bin_acc = []
    bin_count = []

    for b in range(n_bins):
        mask = bin_ids == b
        if mask.sum() == 0:
            bin_conf.append(np.nan)
            bin_acc.append(np.nan)
            bin_count.append(0)
            continue
        bin_conf.append(p[mask].mean())
        bin_acc.append(y_true[mask].mean())
        bin_count.append(int(mask.sum()))

    bin_conf_arr = np.array([x for x in bin_conf if not np.isnan(x)])
    bin_acc_arr = np.array([x for x in bin_acc if not np.isnan(x)])
    bin_count_arr = np.array([c for c in bin_count if c > 0])

    # ECE
    total = max(1, len(p))
    ece = 0.0
    j = 0
    for b in range(n_bins):
        if bin_count[b] == 0:
            continue
        ece += (bin_count[b] / total) * abs(bin_acc[b] - bin_conf[b])
        j += 1

    # Plot reliability
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111)
    ax.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
    ax.plot(bin_conf, bin_acc, marker="o", label=f"Model (ECE={ece:.3f})")
    ax.set_title("Calibration Curve (Reliability Diagram)")
    ax.set_xlabel("Predicted probability (p_yes)")
    ax.set_ylabel("Empirical accuracy")
    ax.legend()
    _save_fig(fig, "24_calibration_curve.png")


# ----------------------------
# Latency plots (needs timings)
# ----------------------------

def has_latency(df: pd.DataFrame) -> bool:
    cols = ["t_l0_llm", "t_l1_retrieval", "t_l2_slm", "t_l3_reasoner", "t_total"]
    return any(c in df.columns and df[c].dropna().size > 0 for c in cols)


def plot_latency_stacked(df: pd.DataFrame):
    if not has_latency(df):
        return

    # Use per-run unique timings (take first row per run)
    g = df.groupby("run_path").first().reset_index()
    # Keep only rows with total
    g = g.dropna(subset=["t_total"])
    if g.empty:
        return

    # Sort by total latency
    g = g.sort_values("t_total", ascending=False).head(25)

    parts = ["t_l0_llm", "t_l1_retrieval", "t_l2_slm", "t_l3_reasoner"]
    parts = [p for p in parts if p in g.columns]

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)

    bottom = np.zeros(len(g))
    x = np.arange(len(g))

    for pcol in parts:
        vals = g[pcol].fillna(0.0).values
        ax.bar(x, vals, bottom=bottom, label=pcol)
        bottom += vals

    ax.set_title("Latency Breakdown per Run (Top 25 by Total Latency)")
    ax.set_xlabel("Run index")
    ax.set_ylabel("Seconds")
    ax.legend()
    ax.set_xticks([])
    _save_fig(fig, "30_latency_stacked.png")


# ----------------------------
# Ablation (needs variant + gold)
# ----------------------------

def plot_ablation_f1(df: pd.DataFrame):
    if not has_gold(df):
        return
    if "variant" not in df.columns or df["variant"].dropna().empty:
        return

    d = df.dropna(subset=["gold_label", "variant", "p_yes"])
    if d.empty:
        return

    # Compute macro F1 per variant based on predicted label vs gold label
    variants = sorted(d["variant"].dropna().unique().tolist())
    labels = _label_order()

    rows = []
    for v in variants:
        dv = d[d["variant"] == v].dropna(subset=["gold_label"])
        if dv.empty:
            continue
        y_true = dv["gold_label"].astype(str)
        y_pred = dv["pred_label"].astype(str)
        _, _, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        rows.append({"variant": v, "f1_macro": f1_macro})

    if not rows:
        return

    rep = pd.DataFrame(rows).sort_values("f1_macro", ascending=False)

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)
    ax.bar(rep["variant"].astype(str), rep["f1_macro"].values)
    ax.set_title("Ablation Study: Macro F1 by Variant")
    ax.set_xlabel("Variant")
    ax.set_ylabel("Macro F1")
    ax.tick_params(axis="x", rotation=25)
    _save_fig(fig, "40_ablation_macro_f1.png")


# ----------------------------
# Error type analysis (needs rule_flag OR gold vs pred mismatch)
# ----------------------------

def plot_error_pie(df: pd.DataFrame):
    # Prefer explicit rule flags
    d = df.dropna(subset=["rule_flag"])
    if d.empty:
        # fallback: if gold exists, use mismatch types
        if not has_gold(df):
            return
        d2 = df.dropna(subset=["gold_label"])
        if d2.empty:
            return
        mism = d2[d2["gold_label"].astype(str) != d2["pred_label"].astype(str)]
        if mism.empty:
            return
        counts = mism["pred_label"].value_counts()
        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111)
        ax.pie(counts.values, labels=counts.index.astype(str), autopct="%1.1f%%")
        ax.set_title("Error Type Pie (by Predicted Label on Gold Mismatches)")
        _save_fig(fig, "50_error_type_pie.png")
        return

    counts = d["rule_flag"].astype(str).value_counts().head(12)
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111)
    ax.pie(counts.values, labels=counts.index.astype(str), autopct="%1.1f%%")
    ax.set_title("Error Type Pie (rule_flag)")
    _save_fig(fig, "50_error_type_pie.png")


# ----------------------------
# Summary CSV
# ----------------------------

def write_summary(df: pd.DataFrame):
    if df.empty:
        return

    by_run = df.groupby("run_path").apply(lambda g: pd.Series({
        "query": g["query"].iloc[0],
        "variant": g["variant"].iloc[0] if "variant" in g.columns else None,
        "n_facts": int(len(g)),
        "n_supported_pred": int((g["pred_label"] == "SUPPORTED").sum()),
        "n_underspecified_pred": int((g["pred_label"] == "UNDER_SPECIFIED").sum()),
        "n_contradicted_pred": int((g["pred_label"] == "CONTRADICTED").sum()),
        "avg_p_yes": float(g["p_yes"].dropna().mean()) if g["p_yes"].dropna().size else np.nan,
        "avg_sim_max": float(g["sim_max"].dropna().mean()) if g["sim_max"].dropna().size else np.nan,
        "avg_uncertainty": float(g["uncertainty"].dropna().mean()) if g["uncertainty"].dropna().size else np.nan,
        "t_total": float(g["t_total"].dropna().iloc[0]) if g["t_total"].dropna().size else np.nan,
    })).reset_index()

    out = OUT_DIR / "summary_metrics.csv"
    by_run.to_csv(out, index=False)
    print(f"[OK] {out}")

    # Also write overall classification report if gold exists
    rep = classification_report_metrics(df)
    if rep is not None:
        out2 = OUT_DIR / "classification_report.csv"
        rep.to_csv(out2, index=False)
        print(f"[OK] {out2}")


def plot_precision_recall_f1_bar(df: pd.DataFrame):
    rep = classification_report_metrics(df)
    if rep is None:
        return

    # Keep only class rows (exclude macro/micro) for a clean plot
    classes = _label_order()
    d = rep[rep["label"].isin(classes)].copy()
    if d.empty:
        return

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)

    x = np.arange(len(classes))
    width = 0.25

    ax.bar(x - width, d["precision"].values, width=width, label="Precision")
    ax.bar(x,          d["recall"].values,    width=width, label="Recall")
    ax.bar(x + width,  d["f1"].values,        width=width, label="F1")

    ax.set_title("Precision / Recall / F1 (per class)")
    ax.set_xlabel("Class")
    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.legend()
    _save_fig(fig, "25_precision_recall_f1_bar.png")


# ----------------------------
# Main
# ----------------------------

def main():
    if _HAS_SEABORN:
        sns.set_style("whitegrid")

    runs = load_runs()
    if not runs:
        return

    df = flatten_runs(runs)
    if df.empty:
        print("[WARN] No per_fact rows found in runs.")
        return

    # Core "must-have" plots (work without gold labels)
    plot_pred_label_bar(df)
    plot_pred_label_stacked_by_query(df, top_n=20)
    plot_hist(df, "p_yes", "Distribution of SLM p_yes", "10_p_yes_hist.png", by_label=False)
    plot_hist(df, "p_yes", "Distribution of SLM p_yes (by predicted label)", "11_p_yes_hist_by_pred.png", by_label=True)
    plot_hist(df, "sim_max", "Distribution of Retrieval Similarity (sim_max)", "12_sim_max_hist.png", by_label=False)
    plot_scatter(df, "sim_max", "p_yes", "sim_max vs p_yes (colored by predicted label)", "13_scatter_sim_vs_pyes.png")

    # Supervised plots (only if gold_label exists)
    plot_confusion_matrix(df)
    plot_precision_recall_f1_bar(df)
    plot_pr_roc(df)
    plot_threshold_vs_f1(df)
    calibration_curve_and_ece(df, n_bins=10)

    # Latency + ablation + error analysis (optional)
    plot_latency_stacked(df)
    plot_ablation_f1(df)
    plot_error_pie(df)

    # Summary CSV(s)
    write_summary(df)

    print("\nDone. Plots saved into outputs/figures/.")


if __name__ == "__main__":
    main()