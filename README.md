# FusionImageEvaluation

A Python toolkit for quantitative evaluation of image fusion methods. Computes 12+ standard metrics for infrared-visible (IR-VI) fusion, multi-focus fusion, medical image fusion, and other fusion tasks. Results are exported to Excel for easy comparison across methods and datasets.

---

## Supported Metrics

| Metric | Description | Better |
|--------|-------------|--------|
| EN | Entropy — information richness | ↑ |
| MI | Mutual Information — source information retention | ↑ |
| SF | Spatial Frequency — detail sharpness | ↑ |
| AG | Average Gradient — edge clarity | ↑ |
| SD | Standard Deviation — contrast | ↑ |
| CC | Correlation Coefficient — structural similarity to sources | ↑ |
| SCD | Sum of Correlations of Differences | ↑ |
| VIF | Visual Information Fidelity | ↑ |
| PSNR | Peak Signal-to-Noise Ratio | ↑ |
| Qabf | Edge information transfer factor | ↑ |
| SSIM | Structural Similarity Index | ↑ |
| MS-SSIM | Multi-Scale SSIM | ↑ |
| PI* | Perceptual Index (no-reference) | ↓ |
| CLIPIQA* | CLIP-based perceptual quality (no-reference) | ↑ |
| MUSIQ* | Multi-scale image quality (no-reference) | ↑ |

\* Optional no-reference metrics, requires `pyiqa`. Enable via `USE_NR_METRICS = True`.

---

## Directory Structure

```
FusionImageEvaluation/
├── Metric_Python/
│   ├── eval_one_method.py   # Batch evaluation entry point
│   ├── eval_one_image.py    # Single image evaluation
│   ├── Metric.py            # All metric implementations
│   ├── Qabf.py              # Qabf implementation
│   ├── Nabf.py              # Nabf implementation
│   ├── FMI.py               # FMI implementation
│   ├── ssim.py              # SSIM / MS-SSIM implementation
│   └── write.py             # Excel write utilities
├── PI/
│   └── pi.py                # Standalone PI (perceptual index) script
├── source_image/
│   └── <DATASET>/
│       ├── ir/              # Infrared source images
│       └── vi/              # Visible source images
├── Results/
│   └── <DATASET>/
│       └── <METHOD>/        # Fusion result images
└── Excel/
    └── <DATASET>/
        └── metric_<METHOD>.xlsx   # Output metrics
```

---

## Installation

```bash
pip install numpy scipy Pillow natsort tqdm openpyxl torch torchvision
# Optional: no-reference metrics
pip install pyiqa
```

---

## Usage

### Batch evaluation (one method, one dataset)

Edit the config at the top of [Metric_Python/eval_one_method.py](Metric_Python/eval_one_method.py):

```python
DATASET = "LLVIP"          # Dataset name, must match source_image/ and Results/ subdirs
METHOD = ["S2Fusion"]      # List of method names to evaluate
USE_NR_METRICS = False     # Set True to also compute PI / CLIPIQA / MUSIQ
NUM_WORKERS = 2            # Parallel workers (set 1 to disable multiprocessing)
```

Then run:

```bash
cd Metric_Python
python eval_one_method.py
```

Results are saved to `Excel/<DATASET>/metric_<METHOD>.xlsx`. The first row of each sheet contains the **mean** across all images.

### Single image evaluation

```bash
cd Metric_Python
python eval_one_image.py
```

Edit the paths at the bottom of the script:

```python
f_name  = 'path/to/fused_image.png'
ir_name = 'path/to/infrared_image.png'
vi_name = 'path/to/visible_image.png'
```

### Perceptual Index (PI) only

```bash
cd PI
python pi.py --input path/to/fused_folder
```

---

## Supported Datasets

The following datasets have been used with this toolkit:

- **LLVIP** — infrared-visible pedestrian detection
- **MSRS** — multi-spectral road scenes
- **FMB** — far-infrared and visible benchmark
- **IVOE** — infrared-visible outdoor evaluation
- **Medical** — medical image fusion (CT/MRI/PET)
- **MFD** — multi-focus defocus dataset
- **MultiFocus** — multi-focus fusion
- **Overexposure** — over-exposed visible image fusion

---

## Adding a New Method

1. Place fusion results under `Results/<DATASET>/<METHOD_NAME>/` with filenames matching the source images.
2. Add the method name to the `METHOD` list in `eval_one_method.py`.
3. Run the script — output goes to `Excel/<DATASET>/metric_<METHOD_NAME>.xlsx`.

---

## About

- Author: warren@伟
- Blog: [CSDN - warren@伟](https://blog.csdn.net/warren103098?type=blog)
