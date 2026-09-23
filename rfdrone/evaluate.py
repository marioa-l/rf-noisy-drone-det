"""Evaluate one or more experiments from their predictions.csv.

    python -m rfdrone.evaluate results/v1_spec_vgg11 results/v1_iq_vgg11 --out results/eval_v1

Writes metrics (mean +- std over folds) as CSV/JSON and publication figures
(PDF + PNG): balanced accuracy vs SNR, accuracy vs SNR, detection (drone vs
noise) balanced accuracy vs SNR, confusion matrices overall and at chosen SNRs.
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix

from .utils import palette, save_fig, save_json, setup_plot_style

NOISE = "Noise"


def load_experiment(path):
    path = Path(path)
    preds = pd.read_csv(path / "predictions.csv")
    summary = json.load(open(path / "summary.json"))
    return preds, summary["class_names"]


def add_detection_columns(df, class_names):
    """Binary drone (1) vs noise (0) labels derived from the 7-class outputs."""
    noise = class_names.index(NOISE)
    df = df.copy()
    df["target_det"] = (df.target != noise).astype(int)
    df["pred_det"] = (df.pred != noise).astype(int)
    return df


def per_fold_metrics(df, group_col=None):
    """Accuracy / balanced accuracy (7-class and detection) per fold, optionally per group."""
    keys = ["fold"] + ([group_col] if group_col else [])
    rows = []
    for key, g in df.groupby(keys):
        key = key if isinstance(key, tuple) else (key,)
        rows.append({**dict(zip(keys, key)), "n": len(g),
                     "acc": accuracy_score(g.target, g.pred),
                     "bal_acc": balanced_accuracy_score(g.target, g.pred),
                     "det_acc": accuracy_score(g.target_det, g.pred_det),
                     "det_bal_acc": balanced_accuracy_score(g.target_det, g.pred_det)})
    return pd.DataFrame(rows)


def summarize(fold_df, group_col=None):
    metrics = ["acc", "bal_acc", "det_acc", "det_bal_acc"]
    if group_col is None:
        agg = fold_df[metrics].agg(["mean", "std"]).T
        agg.columns = ["mean", "std"]
        return agg
    agg = fold_df.groupby(group_col)[metrics].agg(["mean", "std"])
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    return agg.reset_index()


def plot_metric_vs_snr(curves, metric, ylabel, out):
    """curves: list of (name, per-SNR summary DataFrame)."""
    fig, ax = plt.subplots(figsize=(6.5, 4))
    cols = palette(len(curves))
    for (name, s), c in zip(curves, cols):
        ax.errorbar(s["snr"], s[f"{metric}_mean"], yerr=s[f"{metric}_std"].fillna(0), label=name,
                    color=c, marker="o", ms=3.5, lw=1.5, capsize=2)
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1.02)
    ax.axhline(1 / 7, color="grey", ls=":", lw=1, label="chance (7 classes)")
    ax.legend(loc="lower right")
    save_fig(fig, out)


def plot_confusion(df, class_names, title, out):
    cm = confusion_matrix(df.target, df.pred, labels=range(len(class_names)), normalize="true")
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    im = ax.imshow(cm, cmap="viridis", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, label="Row-normalised accuracy")
    ax.set_xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set_xlabel("Prediction")
    ax.set_ylabel("Target")
    ax.set_title(title)
    ax.grid(False)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if cm[i, j] < 0.5 else "black")
    save_fig(fig, out)


def evaluate(exp_paths, names, out_dir, cm_snrs):
    setup_plot_style()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    overall, per_snr_curves, report = [], [], {}
    for path, name in zip(exp_paths, names):
        preds, class_names = load_experiment(path)
        preds = add_detection_columns(preds, class_names)
        folds = per_fold_metrics(preds)
        ov = summarize(folds)
        report[name] = {"n_samples": len(preds), "n_folds": int(preds.fold.nunique()),
                        **{f"{m}_{k}": float(ov.loc[m, k]) for m in ov.index for k in ov.columns}}
        ov.insert(0, "experiment", name)
        overall.append(ov.reset_index().rename(columns={"index": "metric"}))
        snr = summarize(per_fold_metrics(preds, "snr"), "snr")
        snr.to_csv(out_dir / f"{name}_per_snr.csv", index=False)
        summarize(per_fold_metrics(preds, "target"), "target").assign(
            class_name=lambda d: [class_names[t] for t in d.target]).to_csv(
            out_dir / f"{name}_per_class.csv", index=False)
        per_snr_curves.append((name, snr))
        plot_confusion(preds, class_names, f"{name}: all SNRs", out_dir / f"{name}_cm_all")
        for s in cm_snrs:
            sub = preds[preds.snr == s]
            if len(sub):
                plot_confusion(sub, class_names, f"{name}: SNR {s} dB", out_dir / f"{name}_cm_snr{s}")
    pd.concat(overall).to_csv(out_dir / "overall_metrics.csv", index=False)
    save_json(report, out_dir / "overall_metrics.json")
    plot_metric_vs_snr(per_snr_curves, "bal_acc", "Balanced accuracy", out_dir / "bal_acc_vs_snr")
    plot_metric_vs_snr(per_snr_curves, "acc", "Accuracy", out_dir / "acc_vs_snr")
    plot_metric_vs_snr(per_snr_curves, "det_bal_acc", "Detection balanced accuracy",
                       out_dir / "detection_bal_acc_vs_snr")
    print(pd.concat(overall).to_string(index=False))
    return report


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("experiments", nargs="+", help="result directories with predictions.csv")
    ap.add_argument("--names", nargs="*", help="display names (default: directory names)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cm-snrs", type=int, nargs="*", default=[-20, -14, -8, 0])
    a = ap.parse_args(argv)
    names = a.names or [Path(p).name for p in a.experiments]
    evaluate(a.experiments, names, a.out, a.cm_snrs)


if __name__ == "__main__":
    main(sys.argv[1:])
