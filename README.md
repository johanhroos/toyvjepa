# ToyJEPA

AdamW vs. Muon for V-JEPA-style self-supervised video pretraining.

The thesis pretrains the same model on Something-Something-v2 and ImageNet-1k with each optimizer, in pairs of runs that share initial weights and data order. It measures how much compute each optimizer needs to reach the same frozen-encoder linear-probe accuracy on SSv2, how similar the learned representations are (CKA), and whether checkpoints with matched SSv2 accuracy transfer differently to Diving48.

Some differences from V-JEPA 2.1: every block has QK-norm, a per-head sigmoid gate on the attention output and a LayerNorm on each attention and MLP output. These stabilize training, especially using Muon.

## Environment

The project uses [uv](https://docs.astral.sh/uv/). The environment was set up for an RTX5090.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # only if uv is not installed
uv sync
uv run wandb login                                 # training logs to the W&B project `toyjepa`
```

Run Python through `uv run`, e.g. `uv run python -m toyjepa.train`. Set `WANDB_MODE=offline` to train without logging to W&B.

## Ruff

The lint and format settings are in `pyproject.toml`.

```bash
uv run ruff check --fix .    # lint and apply safe fixes
uv run ruff format .         # format
```

## Data

The datasets are not included; each is subject to its own license and terms of use.

Datasets live in `data/` at the repository root (`data/` can be a symlink). The `manifests/` folders are created on the first run. Take care that the directory tree looks like described below.

```text
data/
├── something-something-v2/
│   ├── labels/
│   │   ├── labels.json
│   │   ├── train.json
│   │   └── validation.json
│   ├── videos/
│   │   └── 20bn-something-something-v2/   # <id>.webm
│   └── manifests/
├── ILSVRC/
│   ├── LOC_synset_mapping.txt
│   ├── Data/
│   │   └── CLS-LOC/
│   │       ├── train/                     # <synset>/*.JPEG
│   │       └── val/                       # *.JPEG
│   ├── Annotations/
│   │   └── CLS-LOC/
│   │       ├── train/                     # <synset>/*.xml
│   │       └── val/                       # *.xml
│   └── manifests/
└── diving48/
    ├── Diving48_train.json
    ├── Diving48_test.json
    ├── Diving48_vocab.json
    └── Diving48/
        └── rgb/                           # <vid_name>.mp4
```

**ImageNet-1k.** Download the [ImageNet Object Localization Challenge](https://www.kaggle.com/c/imagenet-object-localization-challenge) data from Kaggle.

**Something-Something-v2.** Download the [SSv2](https://www.qualcomm.com/developer/software/something-something-v-2-dataset) data from qualcomm.

**Diving48.** Download the [Diving48](http://www.svcl.ucsd.edu/projects/resound/dataset.html) data from UCSD website.

## Training run sweep for Adamw and Muon

```bash
uv run python -m toyjepa.train --multirun \
  optim.name=muon,adamw \
  trainer.seed=1 \
  'trainer.run_name="thesis-peri-${optim.name}-lr${optim.lr}-s${trainer.seed}"'
```

## Acknowledgements

Small parts of `toyjepa/models/encoder.py` and `toyjepa/models/predictor.py` are adapted from [V-JEPA 2](https://github.com/facebookresearch/vjepa2).
