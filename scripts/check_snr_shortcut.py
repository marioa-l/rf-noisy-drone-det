"""Check whether the 7-class task can be solved from noise-level cues alone.

Two independent checks, both cheap enough to run while the dataset downloads:

A. Design balance. Reads the file names only, which encode class and SNR, and
   reports the class x SNR sample counts. If a class sits at systematically
   different SNRs, a model can separate it by noise level instead of by signal.
B. Scalar-feature baseline. Takes a stratified subset of samples, extracts a
   handful of cheap summary statistics (total power, crest factor, duty-cycle
   proxy, spectral flatness, occupied bandwidth, ...), and cross-validates a
   random forest on them. Whatever balanced accuracy these features reach is
   reachable without looking at the drone signature at all, so it is the floor
   any CNN result has to be judged against.

    python scripts/check_snr_shortcut.py --data $DATA_ROOT/v2_files \
        --per-class 200 --workers 8 --out results/shortcut
"""
import argparse
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rfdrone.data import DEFAULT_CLASSES, read_class_names
from rfdrone.utils import palette, save_fig, save_json, setup_plot_style

FILE_RE = re.compile(r"IQdata_sample(\d+)_target(\d+)_snr(-?\d+)\.pt$")
FEATURES = ["power", "crest_factor", "duty_proxy", "kurtosis", "spectral_flatness",
            "occupied_bw", "peak_over_median_psd", "spectral_centroid", "spectral_spread"]


def list_samples(data_dir):
    data_dir = Path(data_dir)
    rows = [(str(p.relative_to(data_dir)), *map(int, FILE_RE.match(p.name).groups()))
            for p in data_dir.rglob("IQdata_sample*.pt") if FILE_RE.match(p.name)]
    if not rows:
        raise FileNotFoundError(f"no IQdata_sample*.pt files under {data_dir}")
    return pd.DataFrame(rows, columns=["file", "sample_id", "target", "snr"])


def features_of(path):
    """Summary statistics of one sample. No drone-specific structure is used."""
    d = torch.load(path, map_location="cpu", weights_only=False)
    iq = d["x_iq"].float().numpy()
    z = iq[0] + 1j * iq[1]
    p = np.abs(z) ** 2
    mean_p = p.mean()
    psd = np.fft.fftshift(np.abs(np.fft.fft(z)) ** 2)
    psd = psd / psd.sum()
    freq = np.fft.fftshift(np.fft.fftfreq(len(z)))  # normalised, -0.5 .. 0.5
    centroid = float((psd * freq).sum())
    return [float(mean_p),
            float(p.max() / mean_p),
            float((p > 3 * np.median(p)).mean()),
            float(((p - mean_p) ** 4).mean() / mean_p ** 4),
            float(np.exp(np.mean(np.log(psd + 1e-20))) / psd.mean()),
            float((psd > 3 * np.median(psd)).mean()),
            float(psd.max() / np.median(psd)),
            centroid,
            float(np.sqrt((psd * (freq - centroid) ** 2).sum()))]


def stratified_subset(df, per_class, seed):
    rng = np.random.default_rng(seed)
    keep = []
    for _, g in df.groupby("target"):
        idx = g.index.to_numpy()
        keep.append(rng.choice(idx, size=min(per_class, len(idx)), replace=False))
    return df.loc[np.concatenate(keep)].sort_values("sample_id")


def plot_balance(counts, class_names, out):
    fig, ax = plt.subplots(figsize=(12, 3.8))
    im = ax.imshow(counts.values, cmap="viridis", aspect="auto")
    fig.colorbar(im, ax=ax, label="Samples", pad=0.01)
    ax.set_xticks(range(counts.shape[1]), counts.columns)
    ax.set_yticks(range(counts.shape[0]), [class_names[i] for i in counts.index])
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Class")
    ax.grid(False)
    ax.minorticks_off()
    for i in range(counts.shape[0]):
        for j in range(counts.shape[1]):
            v = counts.values[i, j]
            ax.text(j, i, str(v), ha="center", va="center", fontsize=7.5,
                    color="white" if v < counts.values.max() * 0.6 else "black")
    fig.tight_layout()
    save_fig(fig, out)


def plot_shortcut(per_snr, n_classes, out):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    c = palette(2)
    ax.errorbar(per_snr.snr, per_snr.bal_acc_mean, yerr=per_snr.bal_acc_std, color=c[0], marker="o",
                ms=4, lw=1.8, capsize=2.5, label="Scalar-feature baseline")
    ax.axhline(1 / n_classes, color="grey", ls=":", lw=1.2, label=f"Chance ({n_classes} classes)")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Balanced accuracy")
    ax.set_ylim(0, 1.02)
    ax.legend(loc="upper left")
    fig.tight_layout()
    save_fig(fig, out)


def main(data, per_class, workers, folds, seed, out):
    setup_plot_style()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    class_names = read_class_names(data) if (Path(data) / "class_stats.csv").exists() else DEFAULT_CLASSES
    df = list_samples(data)
    print(f"found {len(df)} sample files, {df.target.nunique()} classes, "
          f"{df.snr.nunique()} SNR levels")

    # --- A. design balance ---------------------------------------------------
    counts = df.pivot_table(index="target", columns="snr", values="file", aggfunc="count").fillna(0).astype(int)
    counts.to_csv(out / "class_x_snr_counts.csv")
    plot_balance(counts, class_names, out / "class_x_snr_counts")
    per_class_snr = df.groupby("target").snr.agg(["mean", "std", "count"])
    per_class_snr.index = [class_names[i] for i in per_class_snr.index]
    per_class_snr.to_csv(out / "snr_per_class.csv")
    print("\nmean SNR per class (dB):")
    print(per_class_snr.round(2).to_string())

    # --- B. scalar-feature baseline ------------------------------------------
    sub = stratified_subset(df, per_class, seed)
    paths = [str(Path(data) / f) for f in sub.file]
    print(f"\nextracting features from {len(paths)} samples with {workers} workers")
    with ProcessPoolExecutor(max_workers=workers) as ex:
        feats = list(ex.map(features_of, paths, chunksize=4))
    X = pd.DataFrame(feats, columns=FEATURES)
    X["snr"] = sub.snr.to_numpy()
    y = sub.target.to_numpy()
    X.assign(target=y).to_csv(out / "features.csv", index=False)

    n_classes = len(np.unique(y))
    rows, importances = [], []
    for name, cols in [("features_only", FEATURES), ("features_plus_snr", FEATURES + ["snr"])]:
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        preds = np.empty_like(y)
        for tr, te in skf.split(X[cols], y):
            rf = RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1)
            rf.fit(X[cols].iloc[tr], y[tr])
            preds[te] = rf.predict(X[cols].iloc[te])
            importances.append(pd.Series(rf.feature_importances_, index=cols))
        sub[f"pred_{name}"] = preds
        det = balanced_accuracy_score((y != class_names.index("Noise")).astype(int),
                                      (preds != class_names.index("Noise")).astype(int))
        rows.append({"input": name, "bal_acc": balanced_accuracy_score(y, preds), "det_bal_acc": det})
        print(f"{name}: balanced accuracy {rows[-1]['bal_acc']:.3f}, "
              f"detection balanced accuracy {det:.3f} (chance {1/n_classes:.3f} / 0.500)")

    per_snr = pd.DataFrame([{"snr": s, "n": len(g),
                             "bal_acc_mean": balanced_accuracy_score(g.target, g.pred_features_only),
                             "bal_acc_std": 0.0}
                            for s, g in sub.groupby("snr")])
    per_snr.to_csv(out / "shortcut_per_snr.csv", index=False)
    plot_shortcut(per_snr, n_classes, out / "shortcut_bal_acc_vs_snr")

    imp = pd.concat(importances, axis=1).mean(axis=1).sort_values(ascending=False)
    imp.to_csv(out / "feature_importance.csv")
    save_json({"n_files_seen": len(df), "n_used": len(sub), "per_class_requested": per_class,
               "class_names": class_names, "overall": rows,
               "feature_importance": imp.to_dict(),
               "snr_per_class": per_class_snr.to_dict()}, out / "summary.json")
    print("\ntop features:", ", ".join(imp.head(4).index))
    print("wrote", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="directory with IQdata_sample*.pt files")
    ap.add_argument("--per-class", type=int, default=200)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/shortcut")
    a = ap.parse_args()
    main(a.data, a.per_class, a.workers, a.folds, a.seed, a.out)
