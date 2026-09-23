"""Hypothetical agreement scenario between the spectrogram (2D) and IQ (1D) models.

Simulates per-sample test predictions of both models with the layout of the v2
dataset (7 classes, 26 SNR levels, stratified 5 folds) from assumed per-class
recall curves plus a shared per-sample difficulty, then plots where the two
models agree and where they disagree.

    python scripts/make_scenario_figures.py [--out results/scenario] [--seed 0]
"""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import balanced_accuracy_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rfdrone.utils import save_fig, save_json, setup_plot_style

CLASSES = ["DJI", "FutabaT14", "FutabaT7", "Graupner", "Noise", "Taranis", "Turnigy"]
COUNTS = [1280, 3472, 801, 801, 8872, 1663, 855]  # Table III, JRFID 2024
SNRS = np.arange(-20, 31, 2)
NOISE = CLASSES.index("Noise")
# assumed recall curves per class: (SNR at mid-transition, ceiling, floor, width in dB)
SPEC = {"DJI": (-12.5, .97, .02, 2.0), "FutabaT14": (-15, .99, .02, 2.0), "FutabaT7": (-16.5, .93, .02, 2.0),
        "Graupner": (-23, .99, .02, 2.0), "Noise": (-14, .995, .92, 3.0), "Taranis": (-22, .99, .02, 2.0),
        "Turnigy": (-16.5, .98, .02, 2.0)}
IQ = {"DJI": (-4.5, .95, .02, 2.5), "FutabaT14": (-7, .97, .02, 2.5), "FutabaT7": (-8.5, .98, .02, 2.5),
      "Graupner": (-15, .98, .02, 2.5), "Noise": (-8, .985, .88, 3.0), "Taranis": (-14, .98, .02, 2.5),
      "Turnigy": (-8.5, .96, .02, 2.5)}
RHO = 0.7  # correlation of the per-sample difficulty seen by both models
SIBLING = {"FutabaT7": "FutabaT14", "FutabaT14": "FutabaT7"}
CATS = ["both_correct", "only_spec", "only_iq", "both_wrong"]
LABELS = ["Both correct", "Only spectrogram correct", "Only IQ correct", "Both wrong"]
COLORS = {"both_correct": "#009E73", "only_spec": "#0072B2", "only_iq": "#E69F00", "both_wrong": "#D55E00"}
REGION_SHADE = {"Both fail": 0.22, "Disagreement": 0.10, "Agreement": 0.0}


def recall(params, target, snr):
    p = np.array([params[CLASSES[t]] for t in target])
    mid, ceil, floor, width = p.T
    return floor + (ceil - floor) / (1 + np.exp(-(snr - mid) / width))


def wrong_label(target, snr, u):
    """Drone errors go mostly to Noise at low SNR and to a similar drone at high SNR."""
    drones = [i for i in range(len(CLASSES)) if i != NOISE]
    out = np.empty_like(target)
    for i, (t, s) in enumerate(zip(target, snr)):
        if t == NOISE:
            out[i] = drones[int(u[i, 0] * len(drones))]
        elif u[i, 1] < (0.9 if s < 0 else 0.3):
            out[i] = NOISE
        else:
            name = CLASSES[t]
            others = [d for d in drones if d != t]
            out[i] = CLASSES.index(SIBLING[name]) if name in SIBLING else others[int(u[i, 0] * len(others))]
    return out


def simulate(seed):
    rng = np.random.default_rng(seed)
    target = np.repeat(np.arange(len(CLASSES)), COUNTS)
    snr = np.concatenate([np.resize(SNRS, n) for n in COUNTS])
    fold = np.concatenate([rng.permutation(np.arange(n) % 5) for n in COUNTS])
    z = rng.standard_normal(len(target))
    u_spec = norm.cdf(RHO * z + np.sqrt(1 - RHO ** 2) * rng.standard_normal(len(z)))
    u_iq = norm.cdf(RHO * z + np.sqrt(1 - RHO ** 2) * rng.standard_normal(len(z)))
    ok_spec, ok_iq = u_spec < recall(SPEC, target, snr), u_iq < recall(IQ, target, snr)
    shared = rng.random((len(z), 2))
    w_spec = wrong_label(target, snr, np.where(rng.random((len(z), 1)) < 0.7, shared, rng.random((len(z), 2))))
    w_iq = wrong_label(target, snr, np.where(rng.random((len(z), 1)) < 0.7, shared, rng.random((len(z), 2))))
    df = pd.DataFrame({"fold": fold, "target": target, "class": [CLASSES[t] for t in target], "snr": snr,
                       "pred_spec": np.where(ok_spec, target, w_spec), "pred_iq": np.where(ok_iq, target, w_iq)})
    df["outcome"] = np.select([ok_spec & ok_iq, ok_spec & ~ok_iq, ~ok_spec & ok_iq], CATS[:3], CATS[3])
    df["agree"] = df.pred_spec == df.pred_iq
    return df


def per_snr_summary(df):
    """Mean +- std over folds; outcome fractions and agreement are class-balanced
    (averaged over classes), so both_correct + only_spec equals the spectrogram
    model's balanced accuracy."""
    bal = df.groupby(["fold", "snr"]).apply(lambda g: pd.Series({
        "bal_acc_spec": balanced_accuracy_score(g.target, g.pred_spec),
        "bal_acc_iq": balanced_accuracy_score(g.target, g.pred_iq)}), include_groups=False)
    frac = (df.groupby(["fold", "snr", "target"]).outcome.value_counts(normalize=True).unstack(fill_value=0)
            .reindex(columns=CATS, fill_value=0))
    frac["agree"] = df.groupby(["fold", "snr", "target"]).agree.mean()
    per_fold = bal.join(frac.groupby(["fold", "snr"]).mean())
    s = per_fold.groupby("snr").agg(["mean", "std"])
    s.columns = [f"{a}_{b}" for a, b in s.columns]
    s = s.reset_index()
    dis, bw = s.only_spec_mean + s.only_iq_mean, s.both_wrong_mean
    s["region"] = np.where((bw >= dis) & (bw > 0.2), "Both fail", np.where(dis >= 0.08, "Disagreement", "Agreement"))
    return s


def region_spans(s):
    spans, start = [], 0
    for i in range(1, len(s) + 1):
        if i == len(s) or s.region[i] != s.region[start]:
            spans.append((s.region[start], s.snr[start] - 1, s.snr[i - 1] + 1))
            start = i
    return spans


def shade(ax, spans, label_y=None):
    for name, lo, hi in spans:
        if REGION_SHADE[name]:
            ax.axvspan(lo, hi, color="grey", alpha=REGION_SHADE[name], lw=0, zorder=0)
        if label_y is not None:
            ax.text((lo + hi) / 2, label_y, name, ha="center", va="bottom", fontsize=12, fontweight="bold")


def plot_snr_figure(s, out):
    spans = region_spans(s)
    fig, (a0, a1) = plt.subplots(2, 1, figsize=(10, 7.5), sharex=True, gridspec_kw={"height_ratios": [1, 1.1]})
    shade(a0, spans, label_y=1.06)
    for key, name in [("spec", "Spectrogram model (2D VGG11_BN)"), ("iq", "IQ model (1D VGG11_BN)")]:
        col = COLORS[f"only_{key}"]
        a0.errorbar(s.snr, s[f"bal_acc_{key}_mean"], yerr=s[f"bal_acc_{key}_std"], color=col, marker="o",
                    ms=4, lw=1.8, capsize=2.5, label=name)
    i = int(np.argmax(s.bal_acc_spec_mean - s.bal_acc_iq_mean))
    x, y_s, y_i = s.snr[i], s.bal_acc_spec_mean[i], s.bal_acc_iq_mean[i]
    a0.annotate("", xy=(x, y_i), xytext=(x, y_s), arrowprops=dict(arrowstyle="<->", color="black", lw=1.2))
    a0.text(x + 0.4, y_s - 0.07, f"gap {y_s - y_i:.2f}", va="center", ha="left", fontsize=11)
    a0.text(x - 0.4, y_s + 0.02, f"{y_s:.2f}", va="bottom", ha="right", fontsize=11, color=COLORS["only_spec"])
    a0.text(x + 0.4, y_i - 0.03, f"{y_i:.2f}", va="top", ha="left", fontsize=11, color=COLORS["only_iq"])
    a0.axhline(1 / 7, color="grey", ls=":", lw=1.2, label="Chance level (7 classes)")
    a0.set_ylabel("Balanced accuracy")
    a0.set_ylim(0, 1.03)
    a0.legend(loc="lower right", fontsize=11, frameon=True)

    shade(a1, spans)
    bottom = np.zeros(len(s))
    for cat, lab in zip(CATS, LABELS):
        a1.bar(s.snr, s[f"{cat}_mean"], bottom=bottom, width=1.6, color=COLORS[cat], label=lab, zorder=2)
        bottom += s[f"{cat}_mean"].values
    a1.errorbar(s.snr, s.agree_mean, yerr=s.agree_std, color="black", marker="s", ms=3.5, lw=1.4, capsize=2,
                label="Agreement rate (same predicted class)", zorder=3)
    a1.set_ylabel("Fraction of test samples\n(class-balanced)")
    a1.set_xlabel("SNR (dB)")
    a1.set_ylim(0, 1.0)
    a1.set_xlim(-21, 31)
    a1.set_xticks(np.arange(-20, 31, 4))
    a1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, fontsize=11, frameon=False)
    fig.tight_layout()
    save_fig(fig, out)


def plot_heatmap(df, s, out):
    tab = df.pivot_table(index="class", columns="snr", values="agree", aggfunc="mean").reindex(CLASSES)
    fig, ax = plt.subplots(figsize=(13, 4.6))
    im = ax.imshow(tab.values, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    fig.colorbar(im, ax=ax, label="Agreement rate\n(same predicted class)", pad=0.01)
    ax.set_xticks(range(len(SNRS)), SNRS)
    ax.set_yticks(range(len(CLASSES)), CLASSES)
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Target class")
    ax.grid(False)
    ax.minorticks_off()
    for i in range(tab.shape[0]):
        for j in range(tab.shape[1]):
            v = tab.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8.5,
                    color="white" if v < 0.6 else "black")
    for name, lo, hi in region_spans(s):
        j0, j1 = (lo + 1 - SNRS[0]) / 2 - 0.5, (hi - 1 - SNRS[0]) / 2 + 0.5
        if j0 > -0.5:
            ax.axvline(j0, color="white", lw=2.5)
        ax.text((j0 + j1) / 2, -0.75, name, ha="center", va="bottom", fontsize=12, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, out)


def main(out, seed):
    setup_plot_style()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    df = simulate(seed)
    s = per_snr_summary(df)
    df.to_csv(out / "scenario_predictions.csv", index=False)
    s.to_csv(out / "scenario_per_snr.csv", index=False)
    save_json({"seed": seed, "rho": RHO, "recall_spec": SPEC, "recall_iq": IQ, "class_counts": dict(zip(CLASSES, COUNTS)),
               "regions": [dict(zip(["region", "snr_from", "snr_to"], (n, lo + 1, hi - 1))) for n, lo, hi in region_spans(s)]},
              out / "scenario_assumptions.json")
    plot_snr_figure(s, out / "agreement_vs_snr")
    plot_heatmap(df, s, out / "agreement_class_x_snr")
    print(s[["snr", "bal_acc_spec_mean", "bal_acc_iq_mean", "only_spec_mean", "only_iq_mean", "both_wrong_mean",
             "agree_mean", "region"]].round(3).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/scenario")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    main(a.out, a.seed)
