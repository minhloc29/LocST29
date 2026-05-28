Spatial Curriculum Learning
============================

This package implements Phases 2-5 of a spatial curriculum learning pipeline.
It is model-agnostic and expects your PyTorch DataLoaders to yield
(features, targets, spot_index) tuples.

Quick start
-----------
from spatial_curriculum.pipeline import SpatialCurriculumPipeline, PipelineConfig
import anndata as ad, numpy as np

# Load your Phase 1 outputs
from spatial_curriculum.utils.spatial import Phase1Results
p1 = Phase1Results(
    spot_mse   = np.load("phase1_spot_mse.npy"),
    coords     = np.load("phase1_coords.npy"),   # shape (N, 2)
    adata_path = "data/sample.h5ad",
    morans_I   = 0.41,
    morans_p   = 0.001,
)

# Build model and data loaders (your own code)
model     = MyTranscriptomicsModel(...)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn   = torch.nn.MSELoss()
# Loaders must yield (features, targets, spot_index) tuples
train_loader = ...
val_loader   = ...
test_loader  = ...

# Run the pipeline
cfg = PipelineConfig(total_epochs=50)
pipe = SpatialCurriculumPipeline(p1, cfg)
report = pipe.run(
    adata          = ad.read_h5ad(p1.adata_path),
    model          = model,
    baseline_model = baseline_model,
    optimizer      = optimizer,
    loss_fn        = loss_fn,
    train_loader   = train_loader,
    val_loader     = val_loader,
    test_loader    = test_loader,
)

Dataset integration (HER2ST + CSCC)
-----------------------------------
This package includes built-in dataset loaders and Phase 1 utilities so you
can use HER2ST or CSCC data directly. Everything lives inside
spatial_curriculum under a neutral namespace.

from spatial_curriculum import (
    CurriculumDataConfig,
    SpatialModelAdapter,
    load_dataset,
    build_slide_loader,
    prepare_phase1,
)

# 1) Build your model and wrap it
model = SpatialModelAdapter(model)

# 2) Load dataset (HER2ST or CSCC)
cfg = CurriculumDataConfig(dataset="her2st", fold=0, data_root="/path/to/data-root")
testset = load_dataset(cfg, train=False)
test_loader = build_slide_loader(testset, slide_index=0)

# 3) Phase 1 (spot MSE + Moran's I) stored inside spatial_curriculum
p1, adata = prepare_phase1(
    cfg=cfg,
    model=model,
    slide_index=0,
    output_dir="./phase1",
    device="cuda",
)

# 4) Run curriculum pipeline
pipe = SpatialCurriculumPipeline(p1, PipelineConfig(total_epochs=50))
report = pipe.run(
    adata=adata,
    model=model,
    baseline_model=model,
    optimizer=optimizer,
    loss_fn=torch.nn.MSELoss(),
    train_loader=test_loader,
    val_loader=test_loader,
    test_loader=test_loader,
)
