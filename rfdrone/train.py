"""Config-driven K-fold training of a spectrogram (2D) or IQ (1D) VGG.

    python -m rfdrone.train configs/v1_spec_vgg11.yaml [--folds 0 1] [--device mps]

Outputs in <results_dir>/<experiment>/:
  fold{k}/model.pt          best-validation state_dict
  fold{k}/history.csv       per-epoch train/val loss, accuracy, balanced accuracy
  fold{k}/predictions.csv   test-fold predictions with softmax probabilities
  fold{k}/embeddings.npy    256-d dense activations of the test samples (optional)
  predictions.csv           all folds concatenated (every sample exactly once)
  summary.json              per-fold and mean +- std test metrics
"""
import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from tqdm import tqdm

from .data import build_dataset, get_splits
from .models import build_model, count_params
from .utils import get_device, load_config, mean_std, save_json, set_seed


def make_loaders(dataset, split, tcfg):
    idx = {k: dataset.index_of(v) for k, v in split.items()}
    nw = tcfg.get("num_workers", 0)
    sampler = None
    if tcfg.get("balanced_sampler", True):
        y = dataset.targets[dataset.keep][idx["train"]]
        counts = np.bincount(y, minlength=len(dataset.class_names)).astype(float)
        w = torch.as_tensor(1.0 / counts[y], dtype=torch.double)
        sampler = WeightedRandomSampler(w, num_samples=len(w), replacement=True)
    common = dict(batch_size=tcfg["batch_size"], num_workers=nw, persistent_workers=nw > 0,
                  pin_memory=tcfg.get("pin_memory", True), prefetch_factor=4 if nw else None)
    return {
        "train": DataLoader(Subset(dataset, idx["train"]), sampler=sampler,
                            shuffle=sampler is None, drop_last=True, **common),
        "val": DataLoader(Subset(dataset, idx["val"]), **common),
        "test": DataLoader(Subset(dataset, idx["test"]), **common),
    }


def make_optimizer(model, tcfg):
    if tcfg.get("optimizer", "adam") == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=tcfg["lr"], betas=(0.9, 0.999),
                               weight_decay=tcfg.get("weight_decay", 0))
    else:
        opt = torch.optim.SGD(model.parameters(), lr=tcfg["lr"], momentum=tcfg.get("momentum", 0.9),
                              weight_decay=tcfg.get("weight_decay", 0))
    sched = None
    if tcfg.get("plateau"):
        p = tcfg["plateau"]
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=p["factor"],
                                                           patience=p["patience"])
    return opt, sched


def autocast_ctx(device, amp):
    """bf16 autocast on CUDA (GH200) when the config asks for it, no-op otherwise."""
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(amp)
    if dtype is None or device.type != "cuda":
        return torch.autocast(device_type=device.type, enabled=False)
    return torch.autocast(device_type="cuda", dtype=dtype)


@torch.no_grad()
def predict(model, loader, device, with_embeddings=False, amp=None):
    model.eval()
    rows, probs, embs = [], [], []
    for x, y, snr, sid in loader:
        x = x.to(device)
        with autocast_ctx(device, amp):
            logits = model(x).float()
        probs.append(torch.softmax(logits, 1).cpu())
        if with_embeddings:
            embs.append(model.embed(x).float().cpu())
        rows.append(torch.stack([sid, y, snr], 1))
    rows, probs = torch.cat(rows).numpy(), torch.cat(probs).numpy()
    df = pd.DataFrame(rows, columns=["sample_id", "target", "snr"])
    df["pred"] = probs.argmax(1)
    df["confidence"] = probs.max(1)
    for c in range(probs.shape[1]):
        df[f"p{c}"] = probs[:, c]
    return df, (torch.cat(embs).numpy() if with_embeddings else None)


def run_epoch(model, loader, device, criterion, optimizer=None, max_batches=None, desc="", amp=None):
    train = optimizer is not None
    model.train(train)
    loss_sum, n, ys, ps = 0.0, 0, [], []
    it = tqdm(loader, desc=desc, leave=False, total=max_batches or len(loader))
    for b, (x, y, _, _) in enumerate(it):
        if max_batches and b >= max_batches:
            break
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.set_grad_enabled(train), autocast_ctx(device, amp):
            logits = model(x)
            loss = criterion(logits, y)
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        loss_sum += loss.item() * len(y)
        n += len(y)
        ys.append(y.cpu())
        ps.append(logits.argmax(1).cpu())
    ys, ps = torch.cat(ys).numpy(), torch.cat(ps).numpy()
    return {"loss": loss_sum / n, "acc": accuracy_score(ys, ps), "bal_acc": balanced_accuracy_score(ys, ps)}


def train_fold(cfg, dataset, split, fold, out_dir, device):
    tcfg = cfg["train"]
    loaders = make_loaders(dataset, split, tcfg)
    model = build_model(cfg["model"]["name"], cfg["model"]["dim"], len(dataset.class_names),
                        in_channels=dataset.in_channels).to(device)
    print(f"fold {fold}: {cfg['model']['name']} dim={cfg['model']['dim']} params={count_params(model):,} "
          f"train/val/test = {len(split['train'])}/{len(split['val'])}/{len(split['test'])}")
    optimizer, scheduler = make_optimizer(model, tcfg)
    criterion = nn.CrossEntropyLoss()
    best, best_state, history = -1.0, None, []
    max_b, amp = tcfg.get("max_batches_per_epoch"), tcfg.get("amp")
    for epoch in range(tcfg["epochs"]):
        t0 = time.time()
        tr = run_epoch(model, loaders["train"], device, criterion, optimizer, max_b, f"f{fold} e{epoch} train", amp)
        va = run_epoch(model, loaders["val"], device, criterion, None, max_b, f"f{fold} e{epoch} val", amp)
        if scheduler:
            scheduler.step(va["loss"])
        rec = {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"], "time_s": time.time() - t0,
               **{f"train_{k}": v for k, v in tr.items()}, **{f"val_{k}": v for k, v in va.items()}}
        history.append(rec)
        print(f"fold {fold} epoch {epoch:3d} | train loss {tr['loss']:.4f} bal_acc {tr['bal_acc']:.3f} | "
              f"val loss {va['loss']:.4f} bal_acc {va['bal_acc']:.3f} | {rec['time_s']:.0f}s")
        if va["bal_acc"] > best:
            best, best_epoch, best_state = va["bal_acc"], epoch, copy.deepcopy(model.state_dict())
        pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)
    model.load_state_dict(best_state)
    torch.save(best_state, out_dir / "model.pt")
    preds, embs = predict(model, loaders["test"], device, tcfg.get("save_embeddings", False), amp)
    preds.insert(1, "fold", fold)
    preds.to_csv(out_dir / "predictions.csv", index=False)
    if embs is not None:
        np.save(out_dir / "embeddings.npy", embs)
    summary = {"fold": fold, "best_epoch": best_epoch, "best_val_bal_acc": best,
               "test_acc": accuracy_score(preds.target, preds.pred),
               "test_bal_acc": balanced_accuracy_score(preds.target, preds.pred),
               "n_params": count_params(model), "epoch_time_s": float(np.mean([h["time_s"] for h in history]))}
    save_json(summary, out_dir / "summary.json")
    print(f"fold {fold} done: test acc {summary['test_acc']:.4f} bal_acc {summary['test_bal_acc']:.4f} "
          f"(best epoch {best_epoch})")
    return summary


def aggregate(exp_dir, n_folds, class_names):
    folds = [json.load(open(exp_dir / f"fold{k}" / "summary.json")) for k in range(n_folds)
             if (exp_dir / f"fold{k}" / "summary.json").exists()]
    preds = pd.concat([pd.read_csv(exp_dir / f"fold{f['fold']}" / "predictions.csv") for f in folds])
    preds.to_csv(exp_dir / "predictions.csv", index=False)
    out = {"folds": folds, "class_names": class_names}
    for key in ["test_acc", "test_bal_acc", "best_epoch"]:
        out[f"{key}_mean"], out[f"{key}_std"] = mean_std([f[key] for f in folds])
    save_json(out, exp_dir / "summary.json")
    print(f"{len(folds)} folds: test acc {out['test_acc_mean']:.4f} +- {out['test_acc_std']:.4f} | "
          f"bal_acc {out['test_bal_acc_mean']:.4f} +- {out['test_bal_acc_std']:.4f}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--folds", type=int, nargs="*", help="subset of folds to train (default: all)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--overwrite", action="store_true", help="retrain folds that already have results")
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    set_seed(cfg.get("seed", 0))
    device = get_device(a.device)
    dataset = build_dataset(cfg)
    splits = get_splits(cfg, dataset)
    exp_dir = Path(cfg.get("results_dir", "results")) / cfg["experiment"]
    exp_dir.mkdir(parents=True, exist_ok=True)
    json.dump(cfg, open(exp_dir / "config.json", "w"), indent=2)
    print(f"device={device} dataset={type(dataset).__name__} n={len(dataset)} "
          f"representation={dataset.representation}/{dataset.spec_mode}")
    for k in (a.folds if a.folds is not None else range(len(splits))):
        out = exp_dir / f"fold{k}"
        if (out / "summary.json").exists() and not a.overwrite:
            print(f"fold {k}: results exist, skipping (use --overwrite)")
            continue
        out.mkdir(exist_ok=True)
        train_fold(cfg, dataset, splits[k], k, out, device)
    aggregate(exp_dir, len(splits), dataset.class_names)


if __name__ == "__main__":
    main(sys.argv[1:])
