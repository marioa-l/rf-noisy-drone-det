"""Use the authors' Zenodo release (VGG11_BN, 5 folds, dataset v2).

Summarise the authors' own test predictions (no data needed):
    python scripts/zenodo_pretrained.py summarize <zenodo_dir> --out results/authors_vgg11_bn

Run a pretrained fold model on a v2 data directory and write predictions in
this project's format (note: ~80% of the v2 samples were in that model's
training set, so this is a pipeline check, not an unbiased evaluation):
    python scripts/zenodo_pretrained.py predict <zenodo_dir> --fold 0 \
        --data $DATA_ROOT/v2_files --out results/authors_vgg11_bn_fold0 [--limit 200]
"""
import argparse
import pickle
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import rfdrone.models as models
from rfdrone.data import V2FileDataset
from rfdrone.train import predict
from rfdrone.utils import get_device, save_json


def load_pretrained(path):
    """The .pth files pickle a `lib.model_VGG2D.VGG` module; alias it to ours."""
    shim = types.ModuleType("lib.model_VGG2D")
    shim.VGG = models.VGG
    sys.modules["lib"], sys.modules["lib.model_VGG2D"] = types.ModuleType("lib"), shim
    obj = torch.load(path, map_location="cpu", weights_only=False)
    model = models.build_model("vgg11_bn", dim=2, num_classes=7)
    model.load_state_dict(obj.state_dict())
    return model


def summarize(zdir, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    frames, class_names = [], None
    for k in range(5):
        r = pickle.load(open(Path(zdir) / f"results_fold{k}.pkl", "rb"))
        class_names = list(r["class_names"])
        frames.append(pd.DataFrame({"fold": k, "sample_id": -1, "target": r["test_targets"].numpy().astype(int),
                                    "snr": r["test_snrs"].numpy().astype(int),
                                    "pred": r["test_predictions"].numpy().astype(int), "confidence": np.nan}))
    preds = pd.concat(frames)
    preds.to_csv(out / "predictions.csv", index=False)
    folds = [{"fold": k, "test_acc": accuracy_score(g.target, g.pred),
              "test_bal_acc": balanced_accuracy_score(g.target, g.pred)} for k, g in preds.groupby("fold")]
    save_json({"folds": folds, "class_names": class_names, "source": "authors' results_fold*.pkl (Zenodo 14065652)",
               "test_bal_acc_mean": float(np.mean([f["test_bal_acc"] for f in folds])),
               "test_bal_acc_std": float(np.std([f["test_bal_acc"] for f in folds], ddof=1))},
              out / "summary.json")
    print("authors' folds:", [round(f["test_bal_acc"], 4) for f in folds])
    print("note: sample_id is unknown for these predictions (not stored in the pickles)")


def run_predict(zdir, fold, data, out, limit, batch_size, device):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    device = get_device(device)
    model = load_pretrained(Path(zdir) / f"model_fold{fold}.pth").to(device)
    ds = V2FileDataset(data, limit=limit).configure("spec", "complex", 1024)
    loader = DataLoader(ds, batch_size=batch_size, num_workers=2)
    preds, _ = predict(model, loader, device)
    preds.insert(1, "fold", fold)
    preds.to_csv(out / "predictions.csv", index=False)
    s = {"folds": [{"fold": fold, "test_acc": accuracy_score(preds.target, preds.pred),
                    "test_bal_acc": balanced_accuracy_score(preds.target, preds.pred)}],
         "class_names": ds.class_names, "source": f"Zenodo model_fold{fold}.pth on {data} (n={len(preds)})"}
    save_json(s, out / "summary.json")
    print(s["folds"][0])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s1 = sub.add_parser("summarize")
    s1.add_argument("zdir")
    s1.add_argument("--out", required=True)
    s2 = sub.add_parser("predict")
    s2.add_argument("zdir")
    s2.add_argument("--fold", type=int, default=0)
    s2.add_argument("--data", required=True)
    s2.add_argument("--out", required=True)
    s2.add_argument("--limit", type=int)
    s2.add_argument("--batch-size", type=int, default=4)
    s2.add_argument("--device", default="auto")
    a = ap.parse_args()
    if a.cmd == "summarize":
        summarize(a.zdir, a.out)
    else:
        run_predict(a.zdir, a.fold, a.data, a.out, a.limit, a.batch_size, a.device)
