import torch

ckpt = torch.load("model/run1_Hist2ST.ckpt", map_location="cpu")

print(ckpt.keys())