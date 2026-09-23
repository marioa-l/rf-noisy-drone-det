# Drone RF detection and classification: spectrogram (2D) vs IQ (1D) CNNs

Re-implementation and per-sample comparison of the two input representations used by
Glüge et al. for drone detection/classification from noisy RF signals:

| Paper | Dataset | Models | Public artefacts |
|---|---|---|---|
| NCTA 2023, *Robust Drone Detection and Classification from RF Signals Using CNNs* | v1: 98,705 vectors of 2×16,384 IQ samples (1.2 ms @ 14 MHz) + precomputed 2×128×128 complex spectrogram, 7 classes, SNR −20…30 dB, ~23 GB, [Kaggle](https://www.kaggle.com/datasets/sgluege/noisy-drone-rf-signal-classification) | VGG11/13/16/19 on **IQ (1D)** and on **spectrogram (2D)** | dataset only (no code, no weights) |
| IEEE JRFID 2024 (arXiv 2406.18624), *Robust Low-Cost Drone Detection and Classification Using CNNs in Low SNR Environments* | v2: 17,744 vectors of 2×1,048,576 IQ samples (74.9 ms @ 14 MHz), 7 classes, SNR −20…30 dB, 125 GB, [Kaggle](https://www.kaggle.com/datasets/sgluege/noisy-drone-rf-signal-classification-v2) | VGG11_BN…VGG19_BN on **spectrogram (2D) only** | [code](https://github.com/sgluege/Robust-Drone-Detection-and-Classification), [VGG11_BN weights + test predictions, 5 folds](https://zenodo.org/records/14065652) |

Important: the JRFID 2024 paper contains no IQ model. The IQ-vs-spectrogram comparison is the
NCTA 2023 paper on dataset v1. This repository implements both models for both datasets with
one code base, so the comparison can be run on v1 locally and on v2 on the HPC.

Classes: DJI, FutabaT14, FutabaT7, Graupner, Noise, Taranis, Turnigy. "Detection" below means
the binary task drone (any of the six) vs Noise, derived from the 7-class outputs.

## Layout

```
rfdrone/            data.py (datasets, STFT, shared CV splits), models.py (VGG 1D/2D),
                    train.py, evaluate.py, compare.py
configs/            one YAML per experiment (v1_*, v2_*, smoke_*)
scripts/            download_data.sh, convert_v1.py, make_synthetic.py, zenodo_pretrained.py
pretrained/         authors' VGG11_BN fold models + their test predictions (from Zenodo)
results/            experiment outputs (predictions.csv, summary.json, figures as PDF+PNG)
```

## Setup

```bash
conda activate rfSyr                     # torch>=2.4 with MPS/CUDA, see requirements.txt
export DATA_ROOT=/Volumes/<external-disk>/rf_drone_data
bash scripts/download_data.sh v1         # needs ~/.kaggle/kaggle.json
python scripts/convert_v1.py $DATA_ROOT/v1_raw/dataset.pt $DATA_ROOT/v1_memmap   # .pt -> npy memmaps
bash scripts/download_data.sh v2         # 125 GB (250 GB free while unzipping)
```

## Run

```bash
# 1. train both models with identical 5-fold splits (the split file is created by the first run)
python -m rfdrone.train configs/v1_spec_vgg11.yaml
python -m rfdrone.train configs/v1_iq_vgg11.yaml
# 2. metrics (mean +- std over folds) and figures: balanced accuracy vs SNR, confusion matrices
python -m rfdrone.evaluate results/v1_spec_vgg11 results/v1_iq_vgg11 --names Spectrogram IQ --out results/eval_v1
# 3. per-sample agreement analysis (by SNR, by class, cross-model confusion, disagreement examples)
python -m rfdrone.compare results/v1_spec_vgg11 results/v1_iq_vgg11 --names Spectrogram IQ \
    --out results/compare_v1 --config configs/v1_spec_vgg11.yaml
```

The same three commands with `configs/v2_*.yaml` reproduce the JRFID 2024 setup (VGG11_BN,
50 epochs, batch 8, Adam 0.005) plus a 1D VGG11_BN on the 2^20-sample IQ vectors.
`--folds 0` trains a single fold; `--device cpu|mps|cuda` overrides auto-detection;
`limit_samples` and `max_batches_per_epoch` in the config shorten runs for debugging.

Authors' artefacts (no dataset needed):

```bash
python scripts/zenodo_pretrained.py summarize pretrained/vgg11_bn_CV5_epochs50_lr0.005_batchsize8 --out results/authors_vgg11_bn
python -m rfdrone.evaluate results/authors_vgg11_bn --names "Authors VGG11_BN" --out results/eval_authors
# run a pretrained fold on v2 data (sanity check of loader + STFT; 80 % of v2 was in its training set)
python scripts/zenodo_pretrained.py predict pretrained/vgg11_bn_CV5_epochs50_lr0.005_batchsize8 --fold 0 --data $DATA_ROOT/v2_files --out results/authors_fold0_on_v2 --limit 500
```

Illustrative scenario for talks (no dataset and no training needed, a few seconds each):

```bash
python scripts/make_scenario_figures.py     # agreement vs SNR, agreement per class x SNR
python scripts/make_example_signals.py      # one example signal per agreement outcome
```

Both write to `results/scenario/`. The predictions they show are simulated from assumed per-class
recall curves, anchored to the published per-SNR results; the assumptions are stored next to the
figures in `scenario_assumptions.json` and `example_signals_cases.json`.

Smoke test without real data:

```bash
python scripts/make_synthetic.py data/synthetic
python -m rfdrone.train configs/smoke_v1_spec.yaml && python -m rfdrone.train configs/smoke_v1_iq.yaml
python -m rfdrone.compare results/smoke/smoke_v1_spec results/smoke/smoke_v1_iq --names Spec IQ --out results/smoke/compare --config configs/smoke_v1_spec.yaml
```

## Outputs

* `results/<exp>/predictions.csv`: one row per sample (every sample is in exactly one test fold):
  `sample_id, fold, target, snr, pred, confidence, p0..p6`.
* `results/<exp>/summary.json`: per-fold and mean ± std accuracy / balanced accuracy, best epoch.
* `results/eval_*/`: `overall_metrics.csv`, `<name>_per_snr.csv`, `<name>_per_class.csv`,
  `bal_acc_vs_snr`, `acc_vs_snr`, `detection_bal_acc_vs_snr`, `<name>_cm_*` (PDF + PNG).
* `results/compare_*/`: `classification_by_snr.csv`, `classification_by_target_name.csv`,
  `detection_by_*.csv` (fractions of both-correct / only-A / only-B / both-wrong, agreement,
  oracle = either model correct), `agreement_class_x_snr` heatmaps, `cross_confusion_snr_le_-10`,
  `disagreements.csv`, `examples/` (spectrogram + IQ plots of disagreeing samples), `summary.json`.

## Implementation notes

* Model input for the spectrogram models is the **complex** STFT (channels Re, Im) scaled by
  1/n_fft, as in the authors' code, not a log-power image. `spec_mode: logpower` switches to a
  1-channel log-magnitude image if the visual branch of the project should be trained on images.
* v1 ships its spectrogram (scipy, Tukey window, nperseg 128); v2 spectrograms are computed on the
  fly with `torch.stft` (Hann, n_fft = hop = 1024, two-sided), identical to the torchaudio
  transform in the authors' repository. The Zenodo VGG11_BN state_dict loads into `rfdrone.models.VGG`
  without remapping (9,358,535 parameters, as in Table VI of the paper).
* The 1D model replaces Conv2d/MaxPool2d/AdaptiveAvgPool2d by their 1D counterparts (NCTA 2023,
  Sec. 3.1); kernel 3, same channel plan, 256-unit dense layer.
* Splits: stratified K-fold on class, then a stratified 20 % validation part of the training fold.
  Both models of a dataset read the same split file, so the comparison is per sample. The authors'
  code uses repeated `train_test_split`, which does not guarantee each sample is tested once.
* Class imbalance (Noise ≈ 50 % of v1, v2) is handled with a class-balanced random sampler, as in
  the authors' v2 code. The loss is unweighted cross-entropy. Model selection uses the validation
  balanced accuracy.
* The authors' Zenodo pickles contain their test predictions but not the sample ids, so their
  predictions cannot be joined to ours per sample; they are only used to reproduce Table VI / Fig. 6.
