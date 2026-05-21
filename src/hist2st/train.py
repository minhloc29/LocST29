import os
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import TensorBoardLogger

from hist2st.config import TrainConfig
from hist2st.data import pk_load
from hist2st.model import build_model
from hist2st.predict import test_model, get_R, cluster
from hist2st.utils.seeds import set_seed


def build_logger(cfg: TrainConfig) -> TensorBoardLogger:
    log_name = ""
    if cfg.zinb > 0:
        cfg.name += "_nb" if cfg.nb == "T" else "_zinb"
        log_name += f"-{cfg.zinb}"
    if cfg.bake > 0:
        cfg.name += "_bake"
        log_name += f"-{cfg.bake}-{cfg.lamb}"
    log_name = f"{cfg.fold}-{cfg.name}-{cfg.tag}{log_name}-{cfg.policy}-{cfg.neighbor}"
    return TensorBoardLogger(cfg.logger, name=log_name)


def run_train(cfg: TrainConfig) -> None:
    set_seed(cfg.seed)

    trainset = pk_load(cfg.fold, "train", False, cfg.data, neighs=cfg.neighbor, prune=cfg.prune)
    train_loader = DataLoader(trainset, batch_size=1, num_workers=0, shuffle=True)
    testset = pk_load(cfg.fold, "test", False, cfg.data, neighs=cfg.neighbor, prune=cfg.prune)
    test_loader = DataLoader(testset, batch_size=1, num_workers=0, shuffle=False)

    label = None
    if cfg.fold in [5, 11, 17, 23, 26, 30] and cfg.data == "her2st":
        label = testset.label[testset.names[0]]

    genes = 171 if cfg.data == "cscc" else 785
    if cfg.data == "cscc":
        cfg.name += "_cscc"

    logger = build_logger(cfg)
    print(logger.name)

    model = build_model(
        tag=cfg.tag,
        genes=genes,
        lr=cfg.lr,
        label=label,
        dropout=cfg.dropout,
        zinb=cfg.zinb,
        nb=(cfg.nb == "T"),
        bake=cfg.bake,
        lamb=cfg.lamb,
        policy=cfg.policy,
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=[cfg.gpu],
        max_epochs=cfg.epochs,
        logger=logger,
        check_val_every_n_epoch=2,
    )

    trainer.fit(model, train_loader, test_loader)

    os.makedirs("./model", exist_ok=True)
    ckpt_path = f"./model/{cfg.fold}-Hist2ST{'_cscc' if cfg.data == 'cscc' else ''}.ckpt"
    model_state = model.state_dict()
    torch_save_path = ckpt_path
    model.cpu()
    model.eval()
    import torch
    torch.save(model_state, torch_save_path)

    pred, gt = test_model(model, test_loader, "cuda")
    R = get_R(pred, gt)[0]
    print("Pearson Correlation:", float(R.mean()))
    if label is not None:
        _, ari = cluster(pred, label)
        print("ARI:", ari)


def main() -> None:
    cfg = TrainConfig.from_args()
    run_train(cfg)
