#!/usr/bin/env bash
# Stage 5: train a diverse set of models on identical folds and measure whether
# their errors are complementary.
#
# Diversity axes: input representation (complex spectrogram, log-power image,
# raw IQ), time-frequency resolution (n_fft 256 / 1024 / 4096), architecture
# (VGG, ResNet, ViT) and per-window power normalisation on or off.
#
#   export DATA_ROOT=$HOME/rf_data
#   bash scripts/run_zoo.sh                 # fold 0, 2100 samples, 5 epochs
#   LIMIT=null EPOCHS=20 FOLDS="0 1 2 3 4" bash scripts/run_zoo.sh
#
# Then:
#   python -m rfdrone.ensemble results/zoo/* --out results/zoo_analysis
set -euo pipefail

SPEC=configs/v2_spec_vgg11_bn_quick.yaml
IQ=configs/v2_iq_vgg11_bn_quick.yaml
FOLDS="${FOLDS:-0}"
LIMIT="${LIMIT:-2100}"
EPOCHS="${EPOCHS:-5}"
BATCH="${BATCH:-16}"

run () {
    local name=$1 cfg=$2; shift 2
    echo; echo "=== ${name} ==="
    python -m rfdrone.train "$cfg" --folds ${FOLDS} --set \
        experiment="${name}" results_dir=results/zoo splits.file=results/zoo/splits.json \
        data.limit_samples=${LIMIT} train.epochs=${EPOCHS} train.batch_size=${BATCH} \
        data.normalize=power "$@"
}

# reference point: the paper's model, and the same model without normalisation
run spec_vgg11bn          "$SPEC"
run spec_vgg11bn_raw      "$SPEC" data.normalize=null
# architecture diversity on the same input
run spec_resnet18         "$SPEC" model.name=timm:resnet18 data.resize="[256, 256]"
run spec_vit_small        "$SPEC" model.name=timm:vit_small_patch16_224 model.img_size=224 \
                                  data.resize="[224, 224]" train.lr=0.0003
# representation diversity
run spec_logpower_vgg11bn "$SPEC" data.spec_mode=logpower
run spec_nfft256          "$SPEC" data.n_fft=256
run spec_nfft4096         "$SPEC" data.n_fft=4096
run iq_vgg11bn            "$IQ"

echo
echo "=== complementarity ==="
python -m rfdrone.ensemble results/zoo/* --out results/zoo_analysis
