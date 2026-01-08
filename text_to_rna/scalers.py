from . import constants
import torch
from text_to_rna.data import read_json
from functools import lru_cache

SCALERS_DIR = "scalers"


@lru_cache(maxsize=None)
def get_norm_factors(modality: str, device: torch.device):
    norm_factors = read_json(f"{SCALERS_DIR}/{modality}_znorm_factors.json")
    mean = torch.tensor(norm_factors["mean"], device=device)
    std = torch.tensor(norm_factors["std"], device=device)
    return mean, std


def log_cpm(counts: torch.Tensor, pseudo_count=constants.PSEUDO_COUNT) -> torch.Tensor:
    total_cts = counts.sum(dim=1, keepdim=True)
    counts = torch.log(counts / total_cts * 1_000_000 + pseudo_count)
    # counts = torch.cat([counts, total_cts.log()], dim=1)
    return counts


def log_cpm_inverse(
    counts: torch.Tensor, pseudo_count=constants.PSEUDO_COUNT, total_cts=1_000_000
) -> torch.Tensor:
    counts = torch.exp(counts)
    counts = counts - pseudo_count
    counts = counts.clip(min=0)
    total_obs_cts = counts.sum(dim=1, keepdim=True)
    counts = counts * total_cts / total_obs_cts
    return counts.clip(min=0).round()


def z_norm(
    modality: str, arr: torch.Tensor, eps=constants.STD_EPS, scale=constants.STD_SCALE
) -> torch.Tensor:
    norm_factors = read_json(f"{SCALERS_DIR}/{modality}_znorm_factors.json")
    mean = torch.Tensor(norm_factors["mean"]).to(arr.device)
    arr = arr - mean
    if scale:
        std = torch.Tensor(norm_factors["std"]).to(arr.device) + eps
        arr = arr / std
    return arr


def z_norm_inverse(
    modality: str, arr: torch.Tensor, eps=constants.STD_EPS, scale=constants.STD_SCALE
) -> torch.Tensor:
    norm_factors = read_json(f"{SCALERS_DIR}/{modality}_znorm_factors.json")
    mean = torch.Tensor(norm_factors["mean"]).to(arr.device)
    if scale:
        std = torch.Tensor(norm_factors["std"]).to(arr.device) + eps
        arr = arr * std
    arr = arr + mean
    return arr


def transform_expression(modality: str, arr: torch.Tensor) -> torch.Tensor:
    if modality == "microarray":
        return z_norm(modality, arr)
    else:
        arr = log_cpm(arr)
        return z_norm(modality, arr)


def inverse_transform_expression(modality: str, arr: torch.Tensor) -> torch.Tensor:
    if modality == "microarray":
        return z_norm_inverse(modality, arr)
    else:
        arr = z_norm_inverse(modality, arr)
        return log_cpm_inverse(arr)
