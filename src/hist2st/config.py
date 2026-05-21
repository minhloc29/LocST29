from __future__ import annotations

from dataclasses import dataclass
import argparse


@dataclass
class TrainConfig:
    gpu: int = 0
    fold: int = 1
    seed: int = 12000
    epochs: int = 350
    name: str = "hist2ST"
    data: str = "her2st"
    logger: str = "../logs/my_logs"
    lr: float = 1e-5
    dropout: float = 0.2
    bake: int = 5
    lamb: float = 0.5
    nb: str = "F"
    zinb: float = 0.25
    prune: str = "Grid"
    policy: str = "mean"
    neighbor: int = 4
    tag: str = "5-7-2-8-4-16-32"

    @classmethod
    def from_args(cls) -> "TrainConfig":
        parser = argparse.ArgumentParser(description="Train Hist2ST model")
        parser.add_argument("--gpu", type=int, default=cls.gpu)
        parser.add_argument("--fold", type=int, default=cls.fold)
        parser.add_argument("--seed", type=int, default=cls.seed)
        parser.add_argument("--epochs", type=int, default=cls.epochs)
        parser.add_argument("--name", type=str, default=cls.name)
        parser.add_argument("--data", type=str, default=cls.data)
        parser.add_argument("--logger", type=str, default=cls.logger)
        parser.add_argument("--lr", type=float, default=cls.lr)
        parser.add_argument("--dropout", type=float, default=cls.dropout)
        parser.add_argument("--bake", type=int, default=cls.bake)
        parser.add_argument("--lamb", type=float, default=cls.lamb)
        parser.add_argument("--nb", type=str, default=cls.nb)
        parser.add_argument("--zinb", type=float, default=cls.zinb)
        parser.add_argument("--prune", type=str, default=cls.prune)
        parser.add_argument("--policy", type=str, default=cls.policy)
        parser.add_argument("--neighbor", type=int, default=cls.neighbor)
        parser.add_argument("--tag", type=str, default=cls.tag)
        args = parser.parse_args()
        return cls(**vars(args))
