from dataset import ViT_HER2ST, ViT_SKIN


def pk_load(
    fold: int,
    mode: str = "train",
    flatten: bool = False,
    dataset: str = "her2st",
    r: int = 4,
    ori: bool = True,
    adj: bool = True,
    prune: str = "Grid",
    neighs: int = 4,
):
    assert dataset in ["her2st", "cscc"]
    if dataset == "her2st":
        return ViT_HER2ST(
            train=(mode == "train"),
            fold=fold,
            flatten=flatten,
            ori=ori,
            neighs=neighs,
            adj=adj,
            prune=prune,
            r=r,
        )
    return ViT_SKIN(
        train=(mode == "train"),
        fold=fold,
        flatten=flatten,
        ori=ori,
        neighs=neighs,
        adj=adj,
        prune=prune,
        r=r,
    )
