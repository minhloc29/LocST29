import torch
from HIST2ST import Hist2ST


def build_model(
    tag: str,
    genes: int,
    lr: float,
    label,
    dropout: float,
    zinb: float,
    nb: bool,
    bake: int,
    lamb: float,
    policy: str,
) -> Hist2ST:
    kernel, patch, depth1, depth2, depth3, heads, channel = map(int, tag.split("-"))
    return Hist2ST(
        depth1=depth1,
        depth2=depth2,
        depth3=depth3,
        n_genes=genes,
        learning_rate=lr,
        label=label,
        kernel_size=kernel,
        patch_size=patch,
        heads=heads,
        channel=channel,
        dropout=dropout,
        zinb=zinb,
        nb=nb,
        bake=bake,
        lamb=lamb,
        policy=policy,
    )


def load_checkpoint(model: Hist2ST, checkpoint_path: str, device: str) -> Hist2ST:
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    return model
