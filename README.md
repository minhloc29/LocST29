# Hist2ST
Spatial Transcriptomics Prediction from Histology jointly through Transformer and Graph Neural Networks.

Hist2ST predicts spot-level gene expression from histology patches by combining convolutional features with spatial transformers and graph neural networks. It supports HER2-positive breast cancer and cutaneous squamous cell carcinoma datasets.

![(Variational) gcn](Workflow.png)

## Repository layout
- [src/hist2st](src/hist2st) package code
- [scripts](scripts) runnable entrypoints
- [data](data) datasets
- [model](model) checkpoints
- [results](results) outputs

## Install (uv)
This repo includes a [pyproject.toml](pyproject.toml) for uv.

```bash
uv venv
uv pip install -e .
```

If you prefer requirements.txt:

```bash
uv pip install -r requirements.txt
```

## Quickstart
Train a model:

```bash
python scripts/train.py --fold 1
```

Batch fold analysis:

```bash
python scripts/batch_fold_analysis.py --folds 1-10
```

Compute difficulty factors:

```bash
python scripts/difficulty_factors.py --folds 1-10
```

Temporal difficulty tracking:

```bash
python scripts/difficulty_tracking.py --fold 1 --epochs 350
```

## Usage (Python)
```python
import torch
from HIST2ST import Hist2ST

model = Hist2ST(
    depth1=2, depth2=8, depth3=4,
    n_genes=785, learning_rate=1e-5,
    kernel_size=5, patch_size=7, fig_size=112,
    heads=16, channel=32, dropout=0.2,
    zinb=0.25, nb=False,
    bake=5, lamb=0.5,
    policy="mean",
)

# patches: [N, 3, W, H]
# coordinates: [N, 2]
# adjacency: [N, N]
pred_expression = model(patches, coordinates, adjacency)  # [N, n_genes]
```

## Data setup
To run [tutorial.ipynb](tutorial.ipynb):

1. Run `download.sh` in [data](data), or clone `https://github.com/almaan/her2st.git` into [data](data).
2. Run `gunzip *.gz` in `data/her2st/data/ST-cnts/`.

## Datasets
- HER2-positive breast tumor ST data: https://github.com/almaan/her2st/
- Cutaneous squamous cell carcinoma 10x Visium data (GSE144240)
- All datasets: https://www.synapse.org/#!Synapse:syn29738084/files/

## Trained models
Trained models are available at Synapse: https://www.synapse.org/#!Synapse:syn29738084/files/

## Citation
```
@article{zengys,
  title={Spatial Transcriptomics Prediction from Histology jointly through Transformer and Graph Neural Networks},
  author={ Yuansong Zeng, Zhuoyi Wei, Weijiang Yu, Rui Yin,  Bingling Li, Zhonghui Tang, Yutong Lu, Yuedong Yang},
  journal={biorxiv},
  year={2021},
  publisher={Cold Spring Harbor Laboratory}
}
```
