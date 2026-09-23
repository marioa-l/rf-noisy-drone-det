"""Complementarity analysis over several models trained on the same folds.

Answers the question the group cares about for metacognition: is there a set of
models whose errors do not coincide, so that a second model is right where the
best one is wrong?

    python -m rfdrone.ensemble results/zoo/* --out results/zoo_analysis

For every pair it reports disagreement and double fault, i.e. how often both are
wrong at once, and it builds the oracle curve: the accuracy reachable if a
perfect selector picked the right model per sample, as models are added greedily.
The gap between the oracle curve and the best single model is the headroom any
error-detection layer can hope to recover.
"""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

from .evaluate import NOISE, load_experiment
from .utils import palette, save_fig, save_json, setup_plot_style


def load_all(paths, names):
    """Wide table: one row per test sample, one prediction column per model."""
    base, class_names = None, None
    probs = {}
    for path, name in zip(paths, names):
        preds, class_names = load_experiment(path)
        cols = preds[["sample_id", "fold", "target", "snr", "pred"]].rename(columns={"pred": name})
        probs[name] = preds.set_index("sample_id")[[c for c in preds.columns if c.startswith("p")
                                                    and c[1:].isdigit()]].sort_index()
        base = cols if base is None else base.merge(cols[["sample_id", name]], on="sample_id", how="inner")
    if base is None or base.empty:
        raise ValueError("no samples shared by all experiments; were they trained on the same splits file?")
    return base, probs, class_names


def pairwise(df, names):
    rows = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            ok_a, ok_b = df[a] == df.target, df[b] == df.target
            rows.append({"model_a": a, "model_b": b,
                         "agreement": float((df[a] == df[b]).mean()),
                         "disagreement": float((df[a] != df[b]).mean()),
                         "double_fault": float((~ok_a & ~ok_b).mean()),
                         "only_a_correct": float((ok_a & ~ok_b).mean()),
                         "only_b_correct": float((~ok_a & ok_b).mean())})
    return pd.DataFrame(rows)


def oracle_curve(df, names):
    """Greedy forward selection on oracle balanced accuracy."""
    chosen, rows, correct = [], [], {n: (df[n] == df.target).to_numpy() for n in names}
    pool, covered = list(names), np.zeros(len(df), bool)
    while pool:
        best = max(pool, key=lambda n: balanced_accuracy_score(df.target, np.where(covered | correct[n],
                                                                                   df.target, -1)))
        covered |= correct[best]
        chosen.append(best)
        pool.remove(best)
        rows.append({"size": len(chosen), "added": best,
                     "oracle_bal_acc": balanced_accuracy_score(df.target, np.where(covered, df.target, -1)),
                     "oracle_acc": float(covered.mean())})
    return pd.DataFrame(rows), chosen


def soft_vote(df, probs, names):
    common = df.sample_id.to_numpy()
    stack = np.mean([probs[n].loc[common].to_numpy() for n in names], axis=0)
    return stack.argmax(1)


def per_model_metrics(df, names, class_names):
    noise = class_names.index(NOISE)
    rows = []
    for n in names:
        per_fold = [balanced_accuracy_score(g.target, g[n]) for _, g in df.groupby("fold")]
        rows.append({"model": n, "bal_acc": balanced_accuracy_score(df.target, df[n]),
                     "bal_acc_std_over_folds": float(np.std(per_fold, ddof=1)) if len(per_fold) > 1 else 0.0,
                     "det_bal_acc": balanced_accuracy_score(df.target != noise, df[n] != noise),
                     "unique_correct": float(((df[n] == df.target) &
                                              ~pd.concat([df[m] == df.target for m in names if m != n],
                                                         axis=1).any(axis=1)).mean())})
    return pd.DataFrame(rows).sort_values("bal_acc", ascending=False)


def plot_oracle(curve, per_model, out):
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    best = per_model.bal_acc.max()
    ax.plot(curve["size"], curve.oracle_bal_acc, marker="o", ms=5, lw=2, color=palette(2)[0],
            label="Oracle over the selected models")
    ax.axhline(best, color=palette(2)[1], ls="--", lw=1.8, label=f"Best single model ({best:.3f})")
    for _, r in curve.iterrows():
        ax.annotate(r.added, (r["size"], r.oracle_bal_acc), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=9, rotation=20)
    ax.set_xlabel("Models combined (greedy order)")
    ax.set_ylabel("Balanced accuracy")
    ax.set_xticks(curve["size"])
    ax.set_ylim(min(best, curve.oracle_bal_acc.min()) - 0.08, 1.02)
    ax.legend(loc="lower right")
    fig.tight_layout()
    save_fig(fig, out)


def plot_per_snr(df, names, out):
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    cols = palette(len(names) + 1)
    correct = {n: (df[n] == df.target).to_numpy() for n in names}
    any_ok = np.any([correct[n] for n in names], axis=0)
    for n, c in zip(names, cols):
        s = [balanced_accuracy_score(g.target, g[n]) for _, g in df.groupby("snr")]
        ax.plot(sorted(df.snr.unique()), s, marker="o", ms=3.5, lw=1.5, color=c, label=n)
    oracle = [balanced_accuracy_score(g.target, np.where(any_ok[g.index], g.target, -1))
              for _, g in df.reset_index(drop=True).groupby("snr")]
    ax.plot(sorted(df.snr.unique()), oracle, marker="s", ms=4, lw=2.2, color="#1B1D21", label="Oracle")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Balanced accuracy")
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=9, ncol=2, loc="lower right")
    fig.tight_layout()
    save_fig(fig, out)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("experiments", nargs="+")
    ap.add_argument("--names", nargs="*")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    setup_plot_style()
    names = a.names or [Path(p).name for p in a.experiments]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    df, probs, class_names = load_all(a.experiments, names)
    print(f"{len(df)} samples shared by {len(names)} models")
    per_model = per_model_metrics(df, names, class_names)
    pairs = pairwise(df, names)
    curve, order = oracle_curve(df, names)
    vote = soft_vote(df, probs, names)

    per_model.to_csv(out / "per_model.csv", index=False)
    pairs.to_csv(out / "pairwise.csv", index=False)
    curve.to_csv(out / "oracle_curve.csv", index=False)
    plot_oracle(curve, per_model, out / "oracle_vs_models")
    plot_per_snr(df, names, out / "bal_acc_vs_snr_all_models")
    df.assign(soft_vote=vote).to_csv(out / "joined_predictions.csv", index=False)

    summary = {"n_samples": len(df), "models": names, "class_names": class_names,
               "best_single": per_model.iloc[0].to_dict(),
               "oracle_all_models": float(curve.oracle_bal_acc.iloc[-1]),
               "headroom": float(curve.oracle_bal_acc.iloc[-1] - per_model.bal_acc.max()),
               "soft_vote_bal_acc": balanced_accuracy_score(df.target, vote),
               "greedy_order": order}
    save_json(summary, out / "summary.json")
    print(per_model.round(4).to_string(index=False))
    print(f"\noracle with all models: {summary['oracle_all_models']:.4f} | "
          f"best single: {per_model.bal_acc.max():.4f} | headroom: {summary['headroom']:.4f}")
    print(f"soft vote: {summary['soft_vote_bal_acc']:.4f}")
    print("\nmost complementary pairs (highest 'only one correct'):")
    pairs["complementary"] = pairs.only_a_correct + pairs.only_b_correct
    print(pairs.sort_values("complementary", ascending=False).head(5).round(4).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1:])
