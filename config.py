import torch

params = {
    "dataset": "openalex",
    "subset_size": 5000,
    "st_dim": 384,
    "semantic_proj_dim": 128,
    "attn_dim": 64,
    "L": 2,
    "alpha": 0.35,
    "beta": 0.25,
    "gamma": 0.20,
    "delta": 0.10,
    "epsilon": 0.10,
    "epochs": 25,
    "lr": 3e-4,
    "weight_decay": 1e-5,
    "top_k": 10,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}
