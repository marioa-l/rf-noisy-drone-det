"""Per-sample agreement analysis between two models tested on identical folds.

    python -m rfdrone.compare results/v1_spec_vgg11 results/v1_iq_vgg11 \
        --names Spectrogram IQ --out results/compare_v1 [--config configs/v1_spec_vgg11.yaml]

For every test sample both models predicted, the pair is classified as
both_correct / only_A / only_B / both_wrong (7-class task) and the same for
the detection task (drone vs noise). Results are broken down by SNR and by
class, written as CSV/JSON, plotted as PDF+PNG, and the disagreeing samples
are listed for inspection. With --config, example disagreements are plotted
(log-power spectrogram + IQ trace) from the raw data.
"""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix

from .data import build_dataset, log_power
from .evaluate import NOISE, add_detection_columns, load_experiment
from .utils import load_config, palette, save_fig, save_json, setup_plot_style

CATS = ["both_correct", "only_A", "only_B", "both_wrong"]


def outcome(ok_a, ok_b):
    return np.select([ok_a & ok_b, ok_a & ~ok_b, ~ok_a & ok_b], CATS[:3], default=CATS[3])


def join_predictions(pa, pb, class_names):
    cols = ["sample_id", "fold", "target", "snr", "pred", "confidence"]
    m = pa[cols].merge(pb[cols], on=["sample_id", "target", "snr"], suffixes=("_A", "_B"))
    if (m.fold_A != m.fold_B).any():
        raise ValueError("models were not tested on the same folds; use a shared splits file")
    noise = class_names.index(NOISE)
    m["ok_A"], m["ok_B"] = m.pred_A == m.target, m.pred_B == m.target
    m["det_ok_A"] = (m.pred_A != noise) == (m.target != noise)
    m["det_ok_B"] = (m.pred_B != noise) == (m.target != noise)
    m["outcome"] = outcome(m.ok_A, m.ok_B)
    m["det_outcome"] = outcome(m.det_ok_A, m.det_ok_B)
    m["agree"] = m.pred_A == m.pred_B
    m["det_agree"] = (m.pred_A != noise) == (m.pred_B != noise)
    return m


def breakdown(m, by, col="outcome"):
    """Fraction of each outcome category per group, plus agreement and oracle rates."""
    g = m.groupby(by)
    out = g[col].value_counts(normalize=True).unstack(fill_value=0).reindex(columns=CATS, fill_value=0)
    out["n"] = g.size()
    out["agree"] = g["agree" if col == "outcome" else "det_agree"].mean()
    out["oracle"] = out["both_correct"] + out["only_A"] + out["only_B"]
    out["acc_A"] = out["both_correct"] + out["only_A"]
    out["acc_B"] = out["both_correct"] + out["only_B"]
    return out.reset_index()


def plot_stacked(bd, by, names, title, out):
    fig, ax = plt.subplots(figsize=(7.5, 4))
    cols = palette(4)
    labels = ["both correct", f"only {names[0]}", f"only {names[1]}", "both wrong"]
    x = np.arange(len(bd))
    bottom = np.zeros(len(bd))
    for cat, lab, c in zip(CATS, labels, cols):
        ax.bar(x, bd[cat], bottom=bottom, label=lab, color=c, width=0.8)
        bottom += bd[cat].values
    ax.set_xticks(x, bd[by], rotation=45 if by != "snr" else 0, ha="right" if by != "snr" else "center")
    ax.set_xlabel("SNR (dB)" if by == "snr" else "Class")
    ax.set_ylabel("Fraction of test samples")
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=4)
    save_fig(fig, out)


def plot_heatmap(m, names, out, col="agree"):
    """Agreement rate per (class, SNR)."""
    tab = m.pivot_table(index="target_name", columns="snr", values=col, aggfunc="mean")
    fig, ax = plt.subplots(figsize=(9, 3.6))
    im = ax.imshow(tab.values, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    fig.colorbar(im, ax=ax, label=f"{names[0]} / {names[1]} agreement")
    ax.set_xticks(range(tab.shape[1]), tab.columns)
    ax.set_yticks(range(tab.shape[0]), tab.index)
    ax.set_xlabel("SNR (dB)")
    ax.grid(False)
    for i in range(tab.shape[0]):
        for j in range(tab.shape[1]):
            v = tab.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if v < 0.5 else "black")
    save_fig(fig, out)


def plot_cross_confusion(m, class_names, names, snr_max, out):
    sub = m[m.snr <= snr_max]
    cm = confusion_matrix(sub.pred_A, sub.pred_B, labels=range(len(class_names)))
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    im = ax.imshow(cm, cmap="viridis")
    fig.colorbar(im, ax=ax, label="Samples")
    ax.set_xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set_xlabel(f"{names[1]} prediction")
    ax.set_ylabel(f"{names[0]} prediction")
    ax.set_title(f"Cross-model confusion, SNR <= {snr_max} dB (n={len(sub)})")
    ax.grid(False)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=7,
                    color="white" if cm[i, j] < cm.max() / 2 else "black")
    save_fig(fig, out)


def plot_examples(m, dataset, class_names, names, out_dir, n_examples, fs=14e6):
    """Log-power spectrogram and IQ trace of disagreeing samples (one per outcome type)."""
    rng = np.random.default_rng(0)
    for cat in ["only_A", "only_B", "both_wrong"]:
        pool = m[m.outcome == cat]
        if pool.empty:
            continue
        rows = pool.sample(min(n_examples, len(pool)), random_state=int(rng.integers(1e6)))
        for _, r in rows.iterrows():
            iq, spec = dataset.raw(int(r.sample_id))
            lp = log_power(torch.from_numpy(np.asarray(spec)))[0].numpy()
            t_ms = np.arange(iq.shape[1]) / fs * 1e3
            fig, (a0, a1) = plt.subplots(1, 2, figsize=(10, 3.4))
            im = a0.imshow(lp, aspect="auto", cmap="viridis", origin="lower",
                           extent=[0, t_ms[-1], -fs / 2e6, fs / 2e6])
            fig.colorbar(im, ax=a0, label="log10 |S|")
            a0.set_xlabel("Time (ms)")
            a0.set_ylabel("Frequency (MHz)")
            a0.grid(False)
            a1.plot(t_ms, iq[0], lw=0.4, label="I")
            a1.plot(t_ms, iq[1], lw=0.4, label="Q", alpha=0.7)
            a1.set_xlabel("Time (ms)")
            a1.set_ylabel("Amplitude (normalised)")
            a1.legend(loc="upper right")
            fig.suptitle(f"id {int(r.sample_id)} | target {class_names[r.target]} | SNR {r.snr} dB | "
                         f"{names[0]}: {class_names[r.pred_A]} ({r.confidence_A:.2f}) | "
                         f"{names[1]}: {class_names[r.pred_B]} ({r.confidence_B:.2f})", fontsize=10)
            save_fig(fig, out_dir / "examples" / f"{cat}_id{int(r.sample_id)}_snr{r.snr}")


def compare(path_a, path_b, names, out_dir, config=None, n_examples=4, low_snr=-10):
    setup_plot_style()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pa, class_names = load_experiment(path_a)
    pb, _ = load_experiment(path_b)
    m = join_predictions(add_detection_columns(pa, class_names), add_detection_columns(pb, class_names),
                         class_names)
    m["target_name"] = [class_names[t] for t in m.target]
    m.to_csv(out_dir / "joined_predictions.csv", index=False)
    m[~m.agree].to_csv(out_dir / "disagreements.csv", index=False)

    tables = {}
    for by in ["snr", "target_name"]:
        for col, tag in [("outcome", "classification"), ("det_outcome", "detection")]:
            bd = breakdown(m, by, col)
            bd.to_csv(out_dir / f"{tag}_by_{by}.csv", index=False)
            tables[f"{tag}_by_{by}"] = bd
            plot_stacked(bd, by, names, f"{tag.capitalize()} outcome by {'SNR' if by == 'snr' else 'class'}",
                         out_dir / f"{tag}_outcome_by_{by}")
    plot_heatmap(m, names, out_dir / "agreement_class_x_snr")
    plot_heatmap(m, names, out_dir / "detection_agreement_class_x_snr", col="det_agree")
    plot_cross_confusion(m, class_names, names, low_snr, out_dir / f"cross_confusion_snr_le_{low_snr}")

    summary = {"models": dict(zip(["A", "B"], names)), "n_samples": len(m),
               "agreement_rate": float(m.agree.mean()), "detection_agreement_rate": float(m.det_agree.mean()),
               "outcome_fractions": m.outcome.value_counts(normalize=True).to_dict(),
               "detection_outcome_fractions": m.det_outcome.value_counts(normalize=True).to_dict(),
               "acc_A": float(m.ok_A.mean()), "acc_B": float(m.ok_B.mean()),
               "oracle_acc": float((m.ok_A | m.ok_B).mean()),
               "mean_confidence_when_disagree": {names[0]: float(m[~m.agree].confidence_A.mean()),
                                                 names[1]: float(m[~m.agree].confidence_B.mean())}}
    save_json(summary, out_dir / "summary.json")
    print(pd.Series(summary["outcome_fractions"]).to_string())
    print(f"agreement {summary['agreement_rate']:.3f} | acc A {summary['acc_A']:.3f} | acc B {summary['acc_B']:.3f} "
          f"| oracle {summary['oracle_acc']:.3f}")
    if config:
        plot_examples(m, build_dataset(load_config(config)), class_names, names, out_dir, n_examples)
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("exp_a")
    ap.add_argument("exp_b")
    ap.add_argument("--names", nargs=2, default=["A", "B"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", help="data config to plot raw examples of disagreements")
    ap.add_argument("--n-examples", type=int, default=4)
    ap.add_argument("--low-snr", type=int, default=-10)
    a = ap.parse_args(argv)
    compare(a.exp_a, a.exp_b, a.names, a.out, a.config, a.n_examples, a.low_snr)


if __name__ == "__main__":
    main(sys.argv[1:])
