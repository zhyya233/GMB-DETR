# GMB-DETR

**Hub-Centered Cross-Scale Context Redistribution and Scale–Density-Adaptive Supervision for Aerial Small-Object Detection**

GMB-DETR is an end-to-end detector developed for small and densely distributed objects in UAV and optical remote-sensing imagery. It is built on the RT-DETR paradigm and introduces a **Gather–Modulate–Broadcast** feature-processing framework together with scale- and density-adaptive training supervision.

The repository is intended for research in UAV surveillance, traffic monitoring, maritime observation, and optical remote-sensing object detection.

---

## Highlights

- **Frequency-enhanced backbone**
  - `HG-WTConv` preserves shallow high-frequency information through hierarchical gated wavelet processing.
  - `SFM-ResLayer` performs spatial-frequency semantic modulation in deeper backbone stages.

- **Gather–Modulate–Broadcast feature redistribution**
  - Multi-scale backbone features are aligned and gathered into a stride-16 central hub.
  - `ODAM` models global and directional context in the shared feature space.
  - The modulated hub feature is broadcast back to four detection scales.

- **Multi-receptive-field enhancement**
  - `MRFEM` refines broadcast features using shared convolutional weights, independent normalization statistics, and learnable multi-branch aggregation.

- **Four-scale RT-DETR decoding**
  - Features at strides 4, 8, 16, and 32 are supplied to the RT-DETR decoder.
  - The additional high-resolution feature level improves tiny-object representation.

- **Scale–density-adaptive supervision**
  - The training objective coordinates localization, classification-quality estimation, and Hungarian matching for small and densely distributed objects.
  - The supervision strategy introduces no additional inference-time computation.

---

## Framework Overview

GMB-DETR contains three principal stages:

1. **Gathering**  
   Backbone features from multiple resolutions are transformed into a shared stride-16 feature space.

2. **Global modulation**  
   The gathered central-hub feature is processed by `ODAM` to capture long-range and cross-axis context.

3. **Broadcasting**  
   The modulated feature is redistributed to strides 4, 8, 16, and 32. `MRFEM` then refines the fused features before RT-DETR decoding.

```text
Input
  │
  ├── Frequency-enhanced backbone
  │     ├── HG-WTConv / HG-ResLayer
  │     └── SFM-ResLayer
  │
  ├── Multi-scale feature gathering
  │     ├── UpSample
  │     └── SPDConv
  │
  ├── Central feature hub
  │     └── ODAM
  │
  ├── Multi-scale broadcasting
  │     └── MRFEM × 4 scales
  │
  └── Four-scale RT-DETR decoder
        └── Predictions
```

---

## Repository Structure

The released implementation is organized under the `codes/` directory. All paths below are relative to the repository root.

```text
GMB-DETR/
├── codes/
│   ├── datasets/          # Dataset YAML configuration files
│   ├── loss/              # GMB-DETR loss and matching implementations
│   ├── nn/                # Model parser, network modules, and custom layers
│   ├── test/              # Validation, inference, and evaluation scripts
│   ├── train/             # Training scripts and experiment entry points
│   └── yaml/              # GMB-DETR architecture YAML files
├── requirements.txt       # Portable project dependencies
├── requirements-full.txt  # Complete reference environment snapshot
├── LICENSE
└── README.md
```

| Content | Repository path |
|---|---|
| Dataset YAML files | `codes/datasets/` |
| Loss functions | `codes/loss/` |
| Model implementation | `codes/nn/` |
| Test and evaluation scripts | `codes/test/` |
| Training scripts | `codes/train/` |
| Model YAML files | `codes/yaml/` |

The original development files were stored under `GMB-DETR-Elsevier/codes/`. The machine-specific prefix is intentionally omitted from the public repository. Do not commit absolute paths such as `/Users/...` or `/home/...` into public scripts or documentation.

---

## Environment

### Reference environment

| Component | Version |
|---|---:|
| Operating system | Linux |
| Python | 3.10.20 |
| PyTorch | 2.11.0 |
| TorchVision | 0.26.0 |
| PyWavelets | 1.8.0 |
| OpenCV | 4.11.0 |
| NumPy | 2.2.6 |
| CUDA runtime packages | 13.0 series |

A complete snapshot of the development environment is provided in `requirements-full.txt`.

> `requirements-full.txt` is intended primarily for environment archival. It may contain platform-specific CUDA packages and local editable-install records and therefore should not be treated as a universally portable dependency file.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/zhyya233/GMB-DETR.git
cd GMB-DETR
```

Replace the placeholders with the final GitHub account and repository name.

### 2. Create a Conda environment

```bash
conda create -n gmb-detr python=3.10 -y
conda activate gmb-detr
```

### 3. Install PyTorch

Install PyTorch and TorchVision builds compatible with the CUDA runtime available on your machine.

The reference environment used:

```bash
python -m pip install torch==2.11.0 torchvision==0.26.0
```

Users with another CUDA version should install the corresponding PyTorch build before installing the remaining dependencies.

### 4. Install project dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 5. Configure the Python search path

Run all commands from the repository root. When required by the local package layout, add the repository to `PYTHONPATH`:

```bash
export PYTHONPATH="$PWD:$PYTHONPATH"
```

### 6. Verify the repository layout

```bash
python - <<'VERIFY_PY'
from pathlib import Path
import torch
import pywt

required_paths = [
    "codes/datasets",
    "codes/loss",
    "codes/nn",
    "codes/test",
    "codes/train",
    "codes/yaml",
]

missing = [path for path in required_paths if not Path(path).exists()]
if missing:
    raise FileNotFoundError(f"Missing required paths: {missing}")

print("PyTorch:", torch.__version__)
print("PyWavelets:", pywt.__version__)
print("GMB-DETR repository structure verified successfully.")
VERIFY_PY
```

---

## Dataset Preparation

Dataset YAML files are stored in:

```text
codes/datasets/
```

GMB-DETR follows the standard Ultralytics detection-dataset format. For example, a file such as `codes/datasets/VisDrone.yaml` may contain:

```yaml
path: /absolute/path/to/dataset
train: train/images
val: valid/images
test: test/images

names:
  0: object
```

The corresponding dataset layout is:

```text
dataset/
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

Each annotation file follows the normalized YOLO format:

```text
class_id x_center y_center width height
```

Before training or evaluation, update the `path` field in the selected dataset YAML to the actual dataset location on your machine.

The framework can be evaluated on UAV and remote-sensing benchmarks such as:

- VisDrone2019-DET
- SeaDronesSeeV2
- RSOD
- NWPU VHR-10

Dataset images and annotations are not distributed with this repository. Follow the license and redistribution policy of each dataset.

---

## Model Configuration

GMB-DETR architecture YAML files are stored in:

```text
codes/yaml/
```

The default model configuration should be referenced with a repository-relative path, for example:

```text
codes/yaml/GMB-DETR.yaml
```

The model uses four decoder input scales:

```text
P2: stride 4
P3: stride 8
P4: stride 16
P5: stride 32
```

The network implementation and parser registration are maintained under `codes/nn/`. The selected model YAML should use the current module names:

```yaml
HG-WTConv
HG-ResLayer
SFM-ResLayer
ODAM
SPDConv
UpSample
MRFEM
RTDETRDecoder
```

Legacy names such as `DA-RFEM`, `DA_RFEM`, `Dynamic_C3RFEM`, and `WT_Conv_Plus` are not used by the current implementation.

The customized loss and matching logic used by GMB-DETR is stored under:

```text
codes/loss/
```

---

## Training

Training entry points are stored in:

```text
codes/train/
```

Run training commands from the repository root so that all relative paths resolve correctly.

### Training-script entry point

```bash
python codes/train/<TRAIN_SCRIPT>.py
```

Replace `<TRAIN_SCRIPT>.py` with the corresponding script in `codes/train/`. The script should reference repository-relative configuration paths:

```text
model: codes/yaml/GMB-DETR.yaml
data:  codes/datasets/<DATASET>.yaml
```

### Ultralytics Python API

When the released training script exposes the standard Ultralytics API, an equivalent configuration is:

```python
from ultralytics import RTDETR

model = RTDETR("codes/yaml/GMB-DETR.yaml")

model.train(
    data="codes/datasets/VisDrone.yaml",
    epochs=350,
    imgsz=640,
    batch=3,
    device=0,
    optimizer="AdamW",
    lr0=2e-4,
    lrf=0.01,
    weight_decay=5e-4,
    cos_lr=True,
    amp=False,
    seed=0,
    workers=8,
    project="runs/GMB-DETR",
    name="train",
)
```

Replace `VisDrone.yaml` with the required dataset configuration file from `codes/datasets/`.

### Ultralytics command line

```bash
yolo detect train \
    model=codes/yaml/GMB-DETR.yaml \
    data=codes/datasets/VisDrone.yaml \
    epochs=350 \
    imgsz=640 \
    batch=3 \
    device=0 \
    optimizer=AdamW \
    lr0=0.0002 \
    lrf=0.01 \
    weight_decay=0.0005 \
    cos_lr=True \
    amp=False \
    seed=0 \
    project=runs/GMB-DETR \
    name=train
```

Adjust the batch size and number of workers according to the available GPU memory and CPU resources. Keep the remaining settings fixed for controlled comparisons.

---

## Validation and Testing

Validation, inference, and evaluation entry points are stored in:

```text
codes/test/
```

Run the required test script from the repository root:

```bash
python codes/test/<TEST_SCRIPT>.py
```

Replace `<TEST_SCRIPT>.py` with the corresponding script in `codes/test/`.

### Ultralytics Python API

```python
from ultralytics import RTDETR

model = RTDETR("runs/GMB-DETR/train/weights/best.pt")

metrics = model.val(
    data="codes/datasets/VisDrone.yaml",
    imgsz=640,
    batch=1,
    device=0,
    split="val",
)

print(metrics)
```

### Ultralytics command line

```bash
yolo detect val \
    model=runs/GMB-DETR/train/weights/best.pt \
    data=codes/datasets/VisDrone.yaml \
    imgsz=640 \
    batch=1 \
    device=0 \
    split=val
```

For fair model comparison, use the same dataset split, input size, confidence settings, post-processing configuration, and evaluation code for every method.

---

## Inference

### Python API

```python
from ultralytics import RTDETR

model = RTDETR("runs/GMB-DETR/train/weights/best.pt")

results = model.predict(
    source="/path/to/image_or_directory",
    imgsz=640,
    conf=0.25,
    device=0,
    save=True,
    project="runs/GMB-DETR",
    name="predict",
)
```

### Command line

```bash
yolo detect predict \
    model=runs/GMB-DETR/train/weights/best.pt \
    source=/path/to/image_or_directory \
    imgsz=640 \
    conf=0.25 \
    device=0 \
    save=True \
    project=runs/GMB-DETR \
    name=predict
```

---

## Export

Export a trained checkpoint to ONNX:

```bash
yolo export \
    model=runs/GMB-DETR/train/weights/best.pt \
    format=onnx \
    imgsz=640 \
    simplify=True
```

Or use the Python API:

```python
from ultralytics import RTDETR

model = RTDETR("runs/GMB-DETR/train/weights/best.pt")
model.export(format="onnx", imgsz=640, simplify=True)
```

Verify custom operators and numerical consistency carefully when exporting to another deployment runtime.

---

## Path Conventions

All public examples use repository-relative paths:

```text
codes/datasets/
codes/loss/
codes/nn/
codes/test/
codes/train/
codes/yaml/
```

Dataset YAML files may still contain local dataset roots. Each user must update those roots before training or evaluation. Avoid committing personal absolute paths into GitHub.

---

## Reproducibility

For controlled experiments, keep the following factors fixed:

- Dataset split and annotation conversion
- Input resolution
- Random seed
- Number of training epochs
- Optimizer and learning-rate schedule
- Batch size and gradient accumulation
- Data augmentation
- Mixed-precision setting
- Confidence threshold and post-processing
- Evaluation implementation
- Hardware and numerical precision used for latency measurement

The principal reference settings are:

```yaml
imgsz: 640
epochs: 350
batch: 3
optimizer: AdamW
lr0: 0.0002
lrf: 0.01
weight_decay: 0.0005
cos_lr: true
amp: false
seed: 0
```

Exact reproducibility across different GPUs, CUDA versions, and PyTorch releases is not guaranteed because some CUDA kernels may be nondeterministic.

---

## Troubleshooting

### `KeyError: 'MRFEM'`

Confirm that:

1. `MRFEM.py` is present in the appropriate module location under `codes/nn/`.
2. `MRFEM` is imported by the model parser provided under `codes/nn/`.
3. `MRFEM` is included in the appropriate `parse_model()` registration.
4. The model YAML uses `MRFEM`, not a legacy module name.

### `KeyError: 'HG-WTConv'`

The YAML-facing aliases must be registered by the parser code under `codes/nn/` before model parsing:

```python
globals()["HG-WTConv"] = HG_WTConv
globals()["HG-ResLayer"] = HG_ResLayer
globals()["SFM-ResLayer"] = SFM_ResLayer
```

### `ModuleNotFoundError: No module named 'pywt'`

Install PyWavelets:

```bash
python -m pip install PyWavelets
```

The Python import name is `pywt`, while the package name is `PyWavelets`.

### Python cannot find the released GMB-DETR modules

Run commands from the repository root and add it to `PYTHONPATH`:

```bash
export PYTHONPATH="$PWD:$PYTHONPATH"
```

Then confirm that the expected source directories exist:

```bash
ls codes/nn codes/loss codes/yaml
```

If the released files are integrated into a local Ultralytics source tree, ensure that the parser and modules from `codes/nn/` are the versions imported at runtime.

### Out-of-memory error

Reduce the batch size:

```text
batch=2
```

or:

```text
batch=1
```

Keep the input resolution and other experimental settings unchanged when performing controlled comparisons.

---

## Pretrained Weights

Pretrained GMB-DETR checkpoints are not embedded directly in the source repository.

A recommended GitHub Releases structure is:

```text
Releases
├── GMB-DETR-VisDrone-best.pt
├── GMB-DETR-SeaDronesSeeV2-best.pt
├── GMB-DETR-RSOD-best.pt
└── GMB-DETR-NWPU-VHR10-best.pt
```

When publishing a checkpoint, document its training split, class count, image size, framework commit, and evaluation protocol.

---

## Citation

A complete BibTeX entry will be provided after the associated paper is publicly available.

Before publication, users should cite the repository and report the exact commit used in their experiments.

---

## Acknowledgements

This project is developed on top of the Ultralytics codebase and the RT-DETR detection framework. We thank their maintainers and the broader open-source computer-vision community.

Third-party datasets, libraries, and evaluation tools remain subject to their respective licenses.

---

## License

The modified Ultralytics source code follows the **GNU Affero General Public License v3.0 (AGPL-3.0)** unless otherwise stated.

Users are responsible for reviewing the licenses of all third-party components, datasets, pretrained weights, and external code included or referenced by this project.

---

## Contact

For implementation, reproducibility, or evaluation questions, please open a GitHub issue.

Include the following information when reporting a problem:

- Operating system
- Python version
- PyTorch and CUDA versions
- GPU model
- Full error traceback
- Model YAML path
- Dataset YAML path
- Command used to reproduce the issue
- Relevant repository commit