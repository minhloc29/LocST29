from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union
import glob
import os

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scprep as scp
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image, ImageFile
from scipy.spatial import distance

from .utils import Phase1Results, prepare_morans_adata, morans_i_scanpy_from_adata
from .utils import move_to_device, flatten_indices, _resolve_data_root, _require_dir, calc_adj
from .analysis import DifficultyFactors

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None


@dataclass
class DataConfig:
    dataset: str = "her2st"
    fold: int = 0
    r: int = 4
    flatten: bool = False
    ori: bool = True
    adj: bool = True
    prune: str = "Grid"
    neighs: int = 4
    data_root: Optional[Path] = None




class Her2STDataset(Dataset):
    """HER2ST dataset loader."""

    def __init__(
        self,
        train: bool = True,
        fold: int = 0,
        r: int = 4,
        flatten: bool = True,
        ori: bool = False,
        adj: bool = False,
        prune: str = "Grid",
        neighs: int = 4,
        data_root: Optional[Path] = None,
    ):
        super().__init__()

        root = _resolve_data_root(data_root)
        cnt_dir = root / "data" / "her2st" / "data" / "ST-cnts"
        img_dir = root / "data" / "her2st" / "data" / "ST-imgs"
        pos_dir = root / "data" / "her2st" / "data" / "ST-spotfiles"
        lbl_dir = root / "data" / "her2st" / "data" / "ST-pat" / "lbl"
        gene_file = root / "data" / "her_hvg_cut_1000.npy"

        _require_dir(cnt_dir, "count")
        _require_dir(img_dir, "image")
        _require_dir(pos_dir, "spotfile")

        self.cnt_dir = str(cnt_dir)
        self.img_dir = str(img_dir)
        self.pos_dir = str(pos_dir)
        self.lbl_dir = str(lbl_dir)
        self.r = 224 // r

        gene_list = list(np.load(gene_file, allow_pickle=True))
        self.gene_list = gene_list
        names = os.listdir(self.cnt_dir)
        names.sort()
        names = [i[:2] for i in names]
        self.train = train
        self.ori = ori
        self.adj = adj

        samples = names[1:33]

        te_names = [samples[fold]]
        tr_names = list(set(samples) - set(te_names))

        if train:
            self.names = tr_names
        else:
            self.names = te_names

        print(te_names)
        print("Loading imgs...")
        self.img_dict = {i: torch.Tensor(np.array(self.get_img(i))) for i in self.names}
        print("Loading metadata...")
        self.meta_dict = {i: self.get_meta(i) for i in self.names}
        self.label = {i: None for i in self.names}
        self.lbl2id = {
            "invasive cancer": 0,
            "breast glands": 1,
            "immune infiltrate": 2,
            "cancer in situ": 3,
            "connective tissue": 4,
            "adipose tissue": 5,
            "undetermined": -1,
        }
        if not train and self.names[0] in ["A1", "B1", "C1", "D1", "E1", "F1", "G2", "H1", "J1"]:
            self.lbl_dict = {i: self.get_lbl(i) for i in self.names}
            idx = self.meta_dict[self.names[0]].index
            lbl = self.lbl_dict[self.names[0]]
            lbl = lbl.loc[idx, :]["label"].values
            self.label[self.names[0]] = lbl
        elif train:
            for i in self.names:
                idx = self.meta_dict[i].index
                if i in ["A1", "B1", "C1", "D1", "E1", "F1", "G2", "H1", "J1"]:
                    lbl = self.get_lbl(i)
                    lbl = lbl.loc[idx, :]["label"].values
                    lbl = torch.Tensor(list(map(lambda j: self.lbl2id[j], lbl)))
                    self.label[i] = lbl
                else:
                    self.label[i] = torch.full((len(idx),), -1)
        self.gene_set = list(gene_list)
        self.exp_dict = {
            i: scp.transform.log(scp.normalize.library_size_normalize(m[self.gene_set].values))
            for i, m in self.meta_dict.items()
        }
        if self.ori:
            self.ori_dict = {i: m[self.gene_set].values for i, m in self.meta_dict.items()}
            self.counts_dict = {}
            for i, m in self.ori_dict.items():
                n_counts = m.sum(1)
                sf = n_counts / np.median(n_counts)
                self.counts_dict[i] = sf
        self.center_dict = {
            i: np.floor(m[["pixel_x", "pixel_y"]].values).astype(int)
            for i, m in self.meta_dict.items()
        }
        self.loc_dict = {i: m[["x", "y"]].values for i, m in self.meta_dict.items()}
        self.adj_dict = {i: calc_adj(m, neighs, prune_tag=prune) for i, m in self.loc_dict.items()}
        self.patch_dict = {}
        self.lengths = [len(i) for i in self.meta_dict.values()]
        self.cumlen = np.cumsum(self.lengths)
        self.id2name = dict(enumerate(self.names))
        self.flatten = flatten

    def __getitem__(self, index):
        slide_id = self.id2name[index]
        im = self.img_dict[slide_id]
        im = im.permute(1, 0, 2)
        exps = self.exp_dict[slide_id]
        if self.ori:
            oris = self.ori_dict[slide_id]
            sfs = self.counts_dict[slide_id]
        centers = self.center_dict[slide_id]
        loc = self.loc_dict[slide_id]
        adj = self.adj_dict[slide_id]
        patches = self.patch_dict.get(slide_id)
        positions = torch.LongTensor(loc)
        patch_dim = 3 * self.r * self.r * 4
        label = self.label[slide_id]
        exps = torch.Tensor(exps)
        if patches is None:
            n_patches = len(centers)
            if self.flatten:
                patches = torch.zeros((n_patches, patch_dim))
            else:
                patches = torch.zeros((n_patches, 3, 2 * self.r, 2 * self.r))
            for i in range(n_patches):
                center = centers[i]
                x, y = center
                patch = im[(x - self.r) : (x + self.r), (y - self.r) : (y + self.r), :]
                if self.flatten:
                    patches[i] = patch.flatten()
                else:
                    patches[i] = patch.permute(2, 0, 1)
            self.patch_dict[slide_id] = patches
        data = [patches, positions, exps]
        if self.adj:
            data.append(adj)
        if self.ori:
            data += [torch.Tensor(oris), torch.Tensor(sfs)]
        data.append(torch.Tensor(centers))
        _ = label
        return data

    def __len__(self):
        return len(self.exp_dict)

    def get_img(self, name):
        pre = f"{self.img_dir}/{name[0]}/{name}"
        fig_name = os.listdir(pre)[0]
        path = f"{pre}/{fig_name}"
        return Image.open(path)

    def get_cnt(self, name):
        path = f"{self.cnt_dir}/{name}.tsv"
        return pd.read_csv(path, sep="\t", index_col=0)

    def get_pos(self, name):
        path = f"{self.pos_dir}/{name}_selection.tsv"
        df = pd.read_csv(path, sep="\t")

        x = df["x"].values
        y = df["y"].values
        x = np.around(x).astype(int)
        y = np.around(y).astype(int)
        spot_id = []
        for i in range(len(x)):
            spot_id.append(str(x[i]) + "x" + str(y[i]))
        df["id"] = spot_id

        return df

    def get_meta(self, name):
        cnt = self.get_cnt(name)
        pos = self.get_pos(name)
        meta = cnt.join((pos.set_index("id")))
        return meta

    def get_lbl(self, name):
        path = f"{self.lbl_dir}/{name}_labeled_coordinates.tsv"
        df = pd.read_csv(path, sep="\t")

        x = df["x"].values
        y = df["y"].values
        x = np.around(x).astype(int)
        y = np.around(y).astype(int)
        
        spot_id = []
        for i in range(len(x)):
            spot_id.append(str(x[i]) + "x" + str(y[i]))
        df["id"] = spot_id
        df.drop("pixel_x", inplace=True, axis=1)
        df.drop("pixel_y", inplace=True, axis=1)
        df.drop("x", inplace=True, axis=1)
        df.drop("y", inplace=True, axis=1)
        df.set_index("id", inplace=True)
        return df


class CSCCDataset(Dataset):
    """CSCC (GSE144240) dataset loader."""

    def __init__(
        self,
        train: bool = True,
        r: int = 4,
        norm: bool = False,
        fold: int = 0,
        flatten: bool = True,
        ori: bool = False,
        adj: bool = False,
        prune: str = "NA",
        neighs: int = 4,
        data_root: Optional[Path] = None,
    ):
        super().__init__()

        root = _resolve_data_root(data_root)
        data_dir = root / "data" / "GSE144240_RAW"
        gene_file = root / "data" / "skin_hvg_cut_1000.npy"

        _require_dir(data_dir, "cscc data")

        self.dir = str(data_dir) + "/"
        self.r = 224 // r

        patients = ["P2", "P5", "P9", "P10"]
        reps = ["rep1", "rep2", "rep3"]
        names = []
        for i in patients:
            for j in reps:
                names.append(i + "_ST_" + j)
        gene_list = list(np.load(gene_file, allow_pickle=True))

        self.ori = ori
        self.adj = adj
        self.norm = norm
        self.train = train
        self.flatten = flatten
        self.gene_list = gene_list
        samples = names
        te_names = [samples[fold]]
        tr_names = list(set(samples) - set(te_names))

        if train:
            self.names = tr_names
        else:
            self.names = te_names

        print(te_names)
        print("Loading imgs...")
        self.img_dict = {i: torch.Tensor(np.array(self.get_img(i))) for i in self.names}
        print("Loading metadata...")
        self.meta_dict = {i: self.get_meta(i) for i in self.names}

        self.gene_set = list(gene_list)
        if self.norm:
            self.exp_dict = {
                i: sc.pp.scale(
                    scp.transform.log(scp.normalize.library_size_normalize(m[self.gene_set].values))
                )
                for i, m in self.meta_dict.items()
            }
        else:
            self.exp_dict = {
                i: scp.transform.log(scp.normalize.library_size_normalize(m[self.gene_set].values))
                for i, m in self.meta_dict.items()
            }
        if self.ori:
            self.ori_dict = {i: m[self.gene_set].values for i, m in self.meta_dict.items()}
            self.counts_dict = {}
            for i, m in self.ori_dict.items():
                n_counts = m.sum(1)
                sf = n_counts / np.median(n_counts)
                self.counts_dict[i] = sf
        self.center_dict = {
            i: np.floor(m[["pixel_x", "pixel_y"]].values).astype(int)
            for i, m in self.meta_dict.items()
        }
        self.loc_dict = {i: m[["x", "y"]].values for i, m in self.meta_dict.items()}
        self.adj_dict = {i: calc_adj(m, neighs, prune_tag=prune) for i, m in self.loc_dict.items()}
        self.patch_dict = {}
        self.lengths = [len(i) for i in self.meta_dict.values()]
        self.cumlen = np.cumsum(self.lengths)
        self.id2name = dict(enumerate(self.names))

    def __getitem__(self, index):
        slide_id = self.id2name[index]
        im = self.img_dict[slide_id].permute(1, 0, 2)

        exps = self.exp_dict[slide_id]
        if self.ori:
            oris = self.ori_dict[slide_id]
            sfs = self.counts_dict[slide_id]
        adj = self.adj_dict[slide_id]
        centers = self.center_dict[slide_id]
        loc = self.loc_dict[slide_id]
        patches = self.patch_dict.get(slide_id)
        positions = torch.LongTensor(loc)
        patch_dim = 3 * self.r * self.r * 4
        exps = torch.Tensor(exps)
        if patches is None:
            n_patches = len(centers)
            if self.flatten:
                patches = torch.zeros((n_patches, patch_dim))
            else:
                patches = torch.zeros((n_patches, 3, 2 * self.r, 2 * self.r))

            for i in range(n_patches):
                center = centers[i]
                x, y = center
                patch = im[(x - self.r) : (x + self.r), (y - self.r) : (y + self.r), :]
                if self.flatten:
                    patches[i] = patch.flatten()
                else:
                    patches[i] = patch.permute(2, 0, 1)
            self.patch_dict[slide_id] = patches
        data = [patches, positions, exps]
        if self.adj:
            data.append(adj)
        if self.ori:
            data += [torch.Tensor(oris), torch.Tensor(sfs)]
        data.append(torch.Tensor(centers))
        return data

    def __len__(self):
        return len(self.exp_dict)

    def get_img(self, name):
        path = glob.glob(self.dir + "*" + name + ".jpg")[0]
        return Image.open(path)

    def get_cnt(self, name):
        path = glob.glob(self.dir + "*" + name + "_stdata.tsv")[0]
        return pd.read_csv(path, sep="\t", index_col=0)

    def get_pos(self, name):
        path = glob.glob(self.dir + "*spot*" + name + ".tsv")[0]
        df = pd.read_csv(path, sep="\t")

        x = df["x"].values
        y = df["y"].values
        x = np.around(x).astype(int)
        y = np.around(y).astype(int)
        spot_id = []
        for i in range(len(x)):
            spot_id.append(str(x[i]) + "x" + str(y[i]))
        df["id"] = spot_id

        return df

    def get_meta(self, name):
        cnt = self.get_cnt(name)
        pos = self.get_pos(name)
        meta = cnt.join(pos.set_index("id"), how="inner")
        return meta


class SpatialModelAdapter(torch.nn.Module):
    """Adapter to make patch/position/adj models compatible with the curriculum."""

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, batch):
        patches, positions, adj = batch
        if adj is not None and adj.ndim == 3 and adj.shape[0] == 1:
            adj = adj.squeeze(0)
        output = self.model(patches, positions, adj)
        if isinstance(output, (list, tuple)):
            return output[0]
        return output


class MultiSlideAdapter(Dataset):
    def __init__(self, base_dataset):
        self.base = base_dataset

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        data = self.base[idx]
        patches, positions, exp, adj = data[0], data[1], data[2], data[3]
        spot_idx = torch.arange(exp.shape[0], dtype=torch.long)
        return (patches, positions, adj), exp, spot_idx, idx
    
        
class SingleSlideAdapter(Dataset):
    """Expose a single slide as (x, y, spot_index)."""

    def __init__(self, base_dataset: Dataset, slide_index: int = 0):
        if not getattr(base_dataset, "adj", False):
            raise ValueError("Dataset must be built with adj=True")
        self.base_dataset = base_dataset
        self.slide_index = slide_index

    def __len__(self) -> int:
        return 1

    def __getitem__(self, idx: int):
        data = self.base_dataset[self.slide_index]
        patches = data[0]
        positions = data[1]
        exp = data[2]
        adj = data[3]
        spot_idx = torch.arange(exp.shape[0], dtype=torch.long)
        x = (patches, positions, adj)
        return x, exp, spot_idx


def build_slide_loader(
    base_dataset: Dataset,
    slide_index: int = 0,
    batch_size: int = 1,
    num_workers: int = 0,
) -> DataLoader:
    slide_dataset = SingleSlideAdapter(base_dataset, slide_index=slide_index)
    return DataLoader(slide_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


def load_dataset(cfg: DataConfig, train: bool) -> Dataset:
    if cfg.dataset not in {"her2st", "cscc"}:
        raise ValueError("dataset must be 'her2st' or 'cscc'")
    if cfg.dataset == "her2st":
        return Her2STDataset(
            train=train,
            fold=cfg.fold,
            r=cfg.r,
            flatten=cfg.flatten,
            ori=cfg.ori,
            adj=cfg.adj,
            prune=cfg.prune,
            neighs=cfg.neighs,
            data_root=cfg.data_root,
        )
    return CSCCDataset(
        train=train,
        fold=cfg.fold,
        r=cfg.r,
        flatten=cfg.flatten,
        ori=cfg.ori,
        adj=cfg.adj,
        prune=cfg.prune,
        neighs=cfg.neighs,
        data_root=cfg.data_root,
    )


def _compute_morans_i(errors: np.ndarray, W_dense: np.ndarray, n_perm: int = 999) -> Tuple[float, float]:
    adata = prepare_morans_adata(n_obs=len(errors), adj=W_dense)
    return morans_i_scanpy_from_adata(adata, errors, n_perms=n_perm)


def compute_phase1_results(
    model: torch.nn.Module,
    loader: DataLoader,
    coords: np.ndarray,
    adj: np.ndarray,
    output_dir: Union[Path, str],
    adata_path: Optional[Union[Path, str]] = None,
    device: Union[str, torch.device] = "cpu",
    n_perm: int = 999,
) -> Tuple[Phase1Results, ad.AnnData]:
    device = torch.device(device)
    model = model.to(device)
    model.eval()

    all_pred, all_target, all_idx = [], [], []
    for x, y, idx in loader:
        x = move_to_device(x, device)
        with torch.no_grad():
            pred = model(x).detach().cpu().numpy()
        target = y.detach().cpu().numpy()
        idx_flat = flatten_indices(idx).cpu().numpy()

        if pred.ndim >= 3 and pred.shape[0] == 1:
            pred = pred.squeeze(0)
        if target.ndim >= 3 and target.shape[0] == 1:
            target = target.squeeze(0)

        all_pred.append(pred)
        all_target.append(target)
        all_idx.append(idx_flat)

    pred_all = np.concatenate(all_pred, axis=0)
    target_all = np.concatenate(all_target, axis=0)
    if pred_all.ndim >= 3 and pred_all.shape[0] == 1:
        pred_all = pred_all.squeeze(0)
    if target_all.ndim >= 3 and target_all.shape[0] == 1:
        target_all = target_all.squeeze(0)
    idx_all = np.concatenate(all_idx, axis=0)

    n_spots = coords.shape[0]
    ordered_pred = np.zeros((n_spots, pred_all.shape[1]), dtype=pred_all.dtype)
    ordered_target = np.zeros((n_spots, target_all.shape[1]), dtype=target_all.dtype)
    ordered_pred[idx_all] = pred_all
    ordered_target[idx_all] = target_all

    spot_mse = np.mean((ordered_pred - ordered_target) ** 2, axis=1)
    morans_i, morans_p = _compute_morans_i(spot_mse, adj, n_perm=n_perm)

    adata = ad.AnnData(ordered_target)
    adata.obsm["spatial"] = coords

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if adata_path is None:
        adata_path = output_dir / "phase1_targets.h5ad"
    else:
        adata_path = Path(adata_path)
    adata.write_h5ad(adata_path)

    p1 = Phase1Results(
        spot_mse=spot_mse,
        coords=coords,
        adata_path=str(adata_path),
        morans_I=float(morans_i),
        morans_p=float(morans_p),
    )
    return p1, adata


def prepare_phase1(
    cfg: DataConfig,
    model: torch.nn.Module,
    slide_index: int = 0,
    output_dir: Union[Path, str] = "./phase1",
    device: Union[str, torch.device] = "cpu",
    n_perm: int = 999,
) -> Tuple[Phase1Results, ad.AnnData]:
    base_dataset = load_dataset(cfg, train=False)
    loader = build_slide_loader(base_dataset, slide_index=slide_index)

    slide_name = base_dataset.names[slide_index]
    coords = base_dataset.center_dict[slide_name]
    adj = np.asarray(base_dataset.adj_dict[slide_name])

    return compute_phase1_results(
        model=model,
        loader=loader,
        coords=coords,
        adj=adj,
        output_dir=output_dir,
        device=device,
        n_perm=n_perm,
    )
