# ToyJEPA

AdamW vs. Muon for V-JEPA-style self-supervised video pretraining.

## Environment

The project uses [uv](https://docs.astral.sh/uv/). `uv sync` installs Python 3.13, PyTorch 2.11 with CUDA 13.0 wheels and the dev tools (ruff). The CUDA wheels need an NVIDIA driver that supports CUDA 13.0.

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

## Learning-rate sweep

Train each optimizer at a few peak LRs on seed 0 and stop early. `trainer.stop_at_epoch` ends the run after that epoch but keeps the LR schedule of the full `trainer.epochs`.

```bash
uv run python -m toyjepa.train --multirun \
  optim.name=adamw,muon \
  optim.lr=6e-4,9e-4,1.2e-3 \
  trainer.seed=0 \
  trainer.stop_at_epoch=4
```

Jobs run one after another and write to `outputs/pretrain/<dd-mm-yyyy>/<HH-MM-SS>_jepasmall_sweep/<optim>-lr<lr>-s<seed>/`. To train selected runs further, resume them with a later `trainer.stop_at_epoch`. `<sweep dir>` is relative to `outputs/pretrain/`; keep the quotes as written.

```bash
uv run python -m toyjepa.train --multirun \
  optim.name=adamw,muon \
  optim.lr=9e-4 \
  trainer.seed=0 \
  trainer.stop_at_epoch=8 \
  'checkpoint.resume_path="<sweep dir>/${optim.name}-lr${optim.lr}-s${trainer.seed}/checkpoints/epoch_004_stablelr.pt"'
```

## Thesis runs

One AdamW/Muon pair per seed (1, 2 and 3) with the defaults in `configs/`:

```bash
uv run python -m toyjepa.train --multirun \
  optim.name=muon,adamw \
  trainer.seed=1 \
  'trainer.run_name="thesis-peri-${optim.name}-lr${optim.lr}-s${trainer.seed}"'
```

Repeat with `trainer.seed=2` and `trainer.seed=3`, changing nothing else.

## Acknowledgements

Small parts of `toyjepa/models/encoder.py` and `toyjepa/models/predictor.py` are adapted from [V-JEPA 2](https://github.com/facebookresearch/vjepa2).
