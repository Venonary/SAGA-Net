# SAGA-Net

**Spectral amplitude–gradient aggregation network for mold-severity grading of maize kernels in variable-speed pushbroom hyperspectral imaging**

This repository contains the reference PyTorch implementation of SAGA-Net, described in the paper

> **Amplitude–gradient spatial–spectral network with cross-speed global aggregation for mold-severity grading of maize kernels in pushbroom hyperspectral imaging**
> Zhenyu Han, Yongli Zhang, Shaozhong Song, Xuan Liu, Tailin Han. *Frontiers in Plant Science* (2026).

It includes the model definition, the augmentation used in the paper (including the uniform row-drop augmentation), training scripts and configurations for the fixed-size and multi-speed experiments, and an inference script that accepts kernel records of any scan height. The model code reproduces the archived implementation that produced the reported results: with the same random seed it creates the same initial weights and produces the same outputs.

## Architecture

Input: one kernel record `X ∈ R^{462×H×128}` (bands × along-track scan rows × cross-track columns). Output: logits for three mold grades.

1. **Dual-stream representation.** The *amplitude* stream is the reflectance `X`. The *gradient* stream is the adjacent-band difference `D_c = X_{c+1} − X_c` (461 bands).
2. **Ordered local aggregation.** Each stream is compressed along the spectral axis into `K = 40` virtual bands by fixed, normalized Gaussian windows:
   `a_{k,c} = exp(−(u_c−μ_k)²/(2σ_k²)) / (Σ_{c'} exp(−(u_{c'}−μ_k)²/(2σ_k²)) + 1e−8)`,
   with `u_c = (c−½)/C`, `μ_k = (k−½)/K`, `σ_k = 0.5/K + 1e−4` and gain `g_k = 1`. These 240 aggregation parameters (2 streams × 3 × 40) are fixed (`requires_grad=False`).
3. **Per-stream projection.** 1×1 convolution (no bias) + BatchNorm + ReLU gives `A` and `G` (`40×H×W` each).
4. **Cross-stream gating.** `A' = A ⊙ [1 + tanh(φ_G(G))]` and `G' = G ⊙ [1 + tanh(φ_A(A))]`, where `φ_A` and `φ_G` are zero-initialized 1×1 convolutions (so both gates start at exactly 1).
5. **Spatial–spectral encoding.** `A'` and `G'` are stacked as 2 channels (`2×40×H×W`) → Conv3d(2→32, 24×5×5) + BN + ReLU → Conv3d(32→64, 16×5×5) + BN + ReLU (no padding; spectral depth 40 → 17 → 2) → MaxPool3d(1×2×2) → `64×2×H'×60`, where `H' = ⌊(H−8)/2⌋`.
6. **Cross-speed global aggregation.** Averaging over the spatial dimensions gives a 128-d feature whatever the record height, followed by a Linear(128→3) classifier (387 parameters).

| Record (speed) | Input `H` | Pooled `H'` |
|---|---|---|
| 1× | 128 | 60 |
| 2× | 64 | 28 |
| 5× | 26 | 9 |
| 10× | 13 | 2 |

Total parameters: **865,155 (865.2K)**, of which 240 are fixed aggregation parameters. The single-stream (amplitude-only) ablation (`use_gradient_stream: false`) has 840,875 parameters. Any height `H ≥ 10` is accepted; the width must be 128.

## Repository layout

```
SAGA-Net/
├── saga_net.py            # model (SAGANet, build_saga_net, parameter counting)
├── augment.py             # fixed-size augmentations, axis-preserving flips, row-drop, subsampled validation views
├── dataset.py             # reader for .npy cubes listed in a CSV file
├── train.py               # fixed-size and multi-speed training protocols
├── predict.py             # inference / evaluation on records of any height
├── configs/
│   ├── fixed_size.yaml
│   └── multi_speed.yaml
├── tests/smoke_test.py    # parameter-count and forward-pass checks (CPU, random data)
├── requirements.txt
└── LICENSE
```

## Installation

```bash
git clone https://github.com/<account>/SAGA-Net.git
cd SAGA-Net
# install PyTorch 2.5.1 for your platform from https://pytorch.org (CUDA 12.1 build used in the paper), then:
pip install -r requirements.txt
python tests/smoke_test.py
```

## Data format

The hyperspectral dataset used in the paper is **not publicly available** (see [Data availability](#data-availability)). To use the code with your own data, provide:

* one `.npy` file per kernel record, holding a float array of shape `462 × H × 128` (bands × along-track rows × cross-track columns; `H × 128 × 462` is also accepted and transposed). Values are radiometrically corrected reflectance `R = (I_S − I_D)/(I_W − I_D)`, and all voxels outside the kernel mask are set to **zero**. No further preprocessing or normalization is applied. In the paper, records are 128 rows at the baseline speed (1×) and 64, 26 and 13 rows at 2×, 5× and 10×;
* a `labels.csv` file with the columns `path,label,split` (and optionally any other columns, e.g. `kernel_id,speed`), where `path` is relative to the CSV file, `label` is `0` = sound, `1` = slight, `2` = severe, and `split` is `train`, `val` or `test`:

```csv
path,label,split,kernel_id,speed
cubes/k0001_1x.npy,0,train,k0001,1x
cubes/k0001_2x.npy,0,train,k0001,2x
cubes/k0002_1x.npy,2,val,k0002,1x
cubes/k0003_10x.npy,1,test,k0003,10x
```

Split by kernel so that all records of one kernel are in the same split.

## Training

### Fixed-size experiment (462 × 128 × 128 inputs)

```bash
python train.py --config configs/fixed_size.yaml --csv data/labels.csv --seed 20260727
```

Adam (learning rate 3×10⁻³, weight decay 10⁻⁶), batch size 8, StepLR (×0.1 every 10 epochs), 50 epochs, automatic mixed precision (AMP). Training augmentation: random flipping, rotation, local erasing and random resized cropping. The checkpoint with the minimum validation cross-entropy is saved to `runs/fixed_size/seed_<seed>/best.pt`. The paper used seeds 20260727, 20260728 and 20260729.

### Multi-speed experiment (train on 1× only, test at every speed)

```bash
# SAGA-Net
python train.py --config configs/multi_speed.yaml --csv data/labels.csv --seed 314159
# SAGA-Net + row-drop
python train.py --config configs/multi_speed.yaml --csv data/labels.csv --seed 314159 --row-drop --output runs/multi_speed_rowdrop
```

Only the 1× (`H = 128`) records of the training kernels are used (`train_filters: {speed: ["1x"]}`; this requires a `speed` column). AdamW (learning rate 3×10⁻⁴, weight decay 10⁻⁴), batch size 2 with 4-step gradient accumulation, at most 80 epochs, cosine annealing, AMP, random column/row flipping with probability 0.5. The checkpoint is chosen by the **validation selection loss**: the mean FP32 cross-entropy over the full-height 1× validation records and their views row-subsampled at nominal factors 2, 5 and 10. Training stops early after 15 epochs without improvement. No higher-speed record is used for training or checkpoint selection. The paper used seeds 314159, 271828 and 161803.

**Row-drop augmentation** (`--row-drop`, `augment.RowDrop`): for each mini-batch, the full height is kept with probability 0.5. Otherwise one stride `r` is drawn uniformly from {2, …, 10}, and each kernel keeps every `r`-th scan row from its own random start in {0, …, r−1}. This gives `462 × ⌈128/r⌉ × 128` cubes; a kernel whose start leaves one row fewer is padded with one zero (background) row at the bottom.

## Inference on variable-height records

```bash
python predict.py --checkpoint runs/multi_speed/seed_314159/best.pt \
    --csv data/labels.csv --split test --group-by speed --output predictions_test.csv
```

Each record is processed at its native height (batch size 1, FP32), and no speed information is given to the model. The output CSV contains the predicted grade and the class probabilities for each record. If labels are available, overall accuracy, macro-F1, quadratic weighted kappa and confusion matrices are printed for all records and for each value of `--group-by` (e.g. per speed).

From Python:

```python
import torch
from saga_net import build_saga_net

model = build_saga_net().eval()
model.load_state_dict(torch.load("best.pt", map_location="cpu")["model_state"])
cube = torch.rand(1, 462, 26, 128)          # e.g. a 5x record
with torch.no_grad():
    grade = model(cube).argmax(1).item()   # 0 sound, 1 slight, 2 severe
```

## Data availability

The hyperspectral dataset of maize kernels used in the paper is not publicly available because it is subject to the data-management requirements of the funded research projects. Requests to access the dataset should be directed to the corresponding author, Tailin Han (hantl@cust.edu.cn).

## Citation

If you use this code, please cite:

```bibtex
@article{han2026saganet,
  title   = {Amplitude--gradient spatial--spectral network with cross-speed global aggregation for mold-severity grading of maize kernels in pushbroom hyperspectral imaging},
  author  = {Han, Zhenyu and Zhang, Yongli and Song, Shaozhong and Liu, Xuan and Han, Tailin},
  journal = {Frontiers in Plant Science},
  year    = {2026},
  volume  = {TBD},
  pages   = {TBD},
  doi     = {TBD}
}
```

## License

Released under the [MIT License](LICENSE).
