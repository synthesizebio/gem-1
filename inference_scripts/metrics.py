import numpy as np
import torch
import torchmetrics
from text_to_rna.data import read_json


def pearson_from_aggregated_moments(n, x, x2, y, y2, xy):
    x_mean = x / n
    x_var = x2 / n - x_mean**2
    y_mean = y / n
    y_var = y2 / n - y_mean**2
    r = (xy / n - x_mean * y_mean) / (x_var * y_var) ** 0.5
    r = torch.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)
    return (r * x_var).sum() / (x_var.sum() + 1e-12)


class HeterogeneityMetrics(torchmetrics.Metric):
    def __init__(self, dim: int, dataset: str):
        super().__init__()
        self.dataset = dataset
        self.num_bins = 10
        self.add_state(
            "cdf_histogram",
            default=torch.tensor(self.num_bins * [0.0]),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "ecdf_histogram",
            default=torch.tensor(self.num_bins * [0.0]),
            dist_reduce_fx="sum",
        )
        self.add_state("n", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("x", default=torch.zeros([dim]), dist_reduce_fx="sum")
        self.add_state("x2", default=torch.zeros([dim]), dist_reduce_fx="sum")
        self.add_state("y", default=torch.zeros([dim]), dist_reduce_fx="sum")
        self.add_state("y2", default=torch.zeros([dim]), dist_reduce_fx="sum")
        self.add_state("xy", default=torch.zeros([dim]), dist_reduce_fx="sum")

    def update(self, *, batch, samples, cdf: callable = None) -> None:
        if cdf is not None:
            hist = torch.histc(
                cdf(batch["expression"]), bins=self.num_bins, min=0, max=1
            )
            self.cdf_histogram += hist

        sorted_x = torch.sort(samples.to("cpu"), dim=0)[0]
        indices = torch.searchsorted(sorted_x.T, batch["expression"].T.to("cpu")).clip(
            0, samples.shape[0] - 1
        )
        ecdf = torch.arange(1, samples.shape[0] + 1, device="cpu") / samples.shape[0]
        ecdf = ecdf[indices].reshape([-1])
        self.ecdf_histogram += torch.histc(
            ecdf.to(self.ecdf_histogram.device), bins=self.num_bins, min=0, max=1
        )

        observed_std = batch["expression"].std(dim=0)
        synthetic_std = samples.std(dim=0)
        self.n += 1
        self.x += observed_std
        self.x2 += observed_std.pow(2)
        self.y += synthetic_std
        self.y2 += synthetic_std.pow(2)
        self.xy += observed_std * synthetic_std

    def compute(self) -> dict[str, torch.tensor]:
        p_hat_model = self.cdf_histogram / self.cdf_histogram.sum()
        p_hat_samples = self.ecdf_histogram / self.ecdf_histogram.sum()
        r = pearson_from_aggregated_moments(
            self.n, self.x, self.x2, self.y, self.y2, self.xy
        )
        return {
            f"{self.dataset}_ece": self.num_bins
            * (p_hat_model - 1 / self.num_bins).abs().mean(),
            f"{self.dataset}_sample_ece": self.num_bins
            * (p_hat_samples - 1 / self.num_bins).abs().mean(),
            f"{self.dataset}_gene_var_pearson": r,
        }


class RegressionMetrics(torchmetrics.Metric):
    # to do replace with dataset
    def __init__(self, dim: int, dataset: str):
        super().__init__()
        self.dataset = dataset
        self.add_state("n", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("x", default=torch.zeros([dim]), dist_reduce_fx="sum")
        self.add_state("x2", default=torch.zeros([dim]), dist_reduce_fx="sum")
        self.add_state("y", default=torch.zeros([dim]), dist_reduce_fx="sum")
        self.add_state("y2", default=torch.zeros([dim]), dist_reduce_fx="sum")
        self.add_state("xy", default=torch.zeros([dim]), dist_reduce_fx="sum")
        self.add_state(
            "sample_pearson", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )
        self.add_state("sample_total", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("mse", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("mae", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("error_count", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, *, observed: torch.Tensor, synthetic: torch.Tensor) -> None:
        # gene Pearson
        self.n += observed.shape[0]
        self.x += observed.sum(dim=0)
        self.x2 += observed.pow(2).sum(dim=0)
        self.y += synthetic.sum(dim=0)
        self.y2 += synthetic.pow(2).sum(dim=0)
        self.xy += (observed * synthetic).sum(dim=0)

        # sample Pearson is per sample and averaged uniformly over all samples
        x = observed - observed.mean(dim=1, keepdim=True)
        y = synthetic - synthetic.mean(dim=1, keepdim=True)
        x_var = (x**2).mean(dim=1)
        y_var = (y**2).mean(dim=1)
        r = (x * y).mean(dim=1) / (x_var * y_var) ** 0.5
        self.sample_pearson += r.nansum()
        self.sample_total += r.shape[0] - r.isnan().sum()

        # error metrics
        self.mae += (observed - synthetic).abs().sum()
        self.mse += (observed - synthetic).pow(2).sum()
        self.error_count += observed.numel()

    def compute(self) -> dict[str, torch.Tensor]:
        gene_pearson = pearson_from_aggregated_moments(
            self.n, self.x, self.x2, self.y, self.y2, self.xy
        )
        return {
            f"{self.dataset}_gene_pearson": gene_pearson,
            f"{self.dataset}_sample_pearson": self.sample_pearson / self.sample_total,
            f"{self.dataset}_mae": self.mae / self.error_count,
            f"{self.dataset}_mse": self.mse / self.error_count,
        }


### https://github.com/toshas/torch-fidelity/blob/master/torch_fidelity/metric_prc.py
def calc_cdist_full(features_1, features_2, batch_size):
    dists = []
    for feat1_batch in features_1.split(batch_size):
        dists_batch = []
        for feat2_batch in features_2.split(batch_size):
            dists_batch.append(torch.cdist(feat1_batch, feat2_batch).cpu())
        dists.append(torch.cat(dists_batch, dim=1))
    return torch.cat(dists, dim=0)


def calculate_precision_recall_full(features_1, features_2, batch_size, neighborhood=3):
    dist_nn_1 = (
        calc_cdist_full(features_1, features_1, batch_size)
        .kthvalue(neighborhood + 1)
        .values
    )
    dist_nn_2 = (
        calc_cdist_full(features_2, features_2, batch_size)
        .kthvalue(neighborhood + 1)
        .values
    )
    dist_2_1 = calc_cdist_full(features_2, features_1, batch_size)
    dist_1_2 = dist_2_1.T
    # Precision
    precision = (dist_2_1 <= dist_nn_1).any(dim=1).float().mean().item()
    # Recall
    recall = (dist_1_2 <= dist_nn_2).any(dim=1).float().mean().item()
    return precision, recall


### end https://github.com/toshas/torch-fidelity/blob/master/torch_fidelity/metric_prc.py


class RaDMetrics(torchmetrics.Metric):
    def __init__(self, dim_latent: int, dataset: str, max_samples: int = 20_000):
        super().__init__()
        self.dataset = dataset
        self.max_samples = max_samples
        self.add_state("i", default=torch.tensor(0), dist_reduce_fx=None)
        self.add_state(
            "obs_samples",
            default=torch.zeros((max_samples, dim_latent)),
            dist_reduce_fx=None,
        )
        self.add_state(
            "syn_samples",
            default=torch.zeros((max_samples, dim_latent)),
            dist_reduce_fx=None,
        )

    def update(self, *, observed: torch.Tensor, synthetic: torch.Tensor) -> None:
        num_samples = min(self.max_samples - self.i, observed.shape[0])
        self.obs_samples[self.i : self.i + num_samples] = observed[:num_samples]
        self.syn_samples[self.i : self.i + num_samples] = synthetic[:num_samples]
        self.i += num_samples

    def compute(self) -> dict[str, torch.tensor]:
        if self.i == 0:
            return {
                f"{self.dataset}_F1": torch.nan,
                f"{self.dataset}_FID": torch.nan,
                f"{self.dataset}_precision": torch.nan,
                f"{self.dataset}_recall": torch.nan,
            }

        obs_samples = self.obs_samples[: self.i]
        syn_samples = self.syn_samples[: self.i]
        obs_mu = obs_samples.mean(dim=0)
        obs_sigma = torch.cov(obs_samples.T)
        syn_mu = syn_samples.mean(dim=0)
        syn_sigma = torch.cov(syn_samples.T)
        fid = (obs_mu - syn_mu).pow(2).sum() + obs_sigma.trace() + syn_sigma.trace()
        fid -= 2 * torch.linalg.eigvals(obs_sigma @ syn_sigma).sqrt().real.sum()
        precision, recall = calculate_precision_recall_full(
            obs_samples, syn_samples, batch_size=obs_samples.shape[0]
        )
        return {
            f"{self.dataset}_F1": 2
            * precision
            * recall
            / max(1e-5, precision + recall),
            f"{self.dataset}_FID": fid,
            f"{self.dataset}_precision": precision,
            f"{self.dataset}_recall": recall,
        }


class PerturbationMetrics(torchmetrics.Metric):
    # need modality and datasets
    def __init__(self, modality: str, dataset: str):
        super().__init__()
        self.modality = modality
        self.dataset = dataset
        gene_order = read_json("gene_id_order.json")
        if modality in ["czi", "perturbseq"]:
            genes = gene_order["single_cell"]
        else:
            genes = gene_order[modality]

        self.gene_order = genes
        self.perturbation_types = [
            "crispr",
            "overexpression",
            "shrna",
        ]  # You might need to adjust this based on how perturbation types are encoded now

        for perturbation_type in self.perturbation_types:
            self.add_state(
                f"{perturbation_type}_bias",
                default=torch.tensor(0.0),
                dist_reduce_fx="sum",
            )
            self.add_state(
                f"{perturbation_type}_count",
                default=torch.tensor(0.0),
                dist_reduce_fx="sum",
            )

    def update(
        self, *, batch: dict, observed: torch.Tensor, synthetic: torch.Tensor
    ) -> None:
        perturbed_gene_indices = batch.get("modality_specific_perturbed_gene_idx")

        if perturbed_gene_indices is None:
            return  # No perturbation indices found, skip this batch

        # Assuming "perturbation_type" still exists, adjust if needed
        if "perturbation_type" in batch["metadata"]:
            perturbation_types = np.array(batch["metadata"]["perturbation_type"])
        else:
            return  # If perturbation_type is unavailable, metrics can't be calculated

        for perturbation_type in self.perturbation_types:
            mask = np.where(perturbation_types == perturbation_type)[0]
            for i in mask:  # Iterate over samples with this perturbation type
                gene_idx = perturbed_gene_indices[i].item()
                if 0 <= gene_idx < len(self.gene_order):  # Valid index
                    current_bias = getattr(self, f"{perturbation_type}_bias")
                    current_count = getattr(self, f"{perturbation_type}_count")
                    setattr(
                        self,
                        f"{perturbation_type}_bias",
                        current_bias + (synthetic[i, gene_idx] - observed[i, gene_idx]),
                    )
                    setattr(self, f"{perturbation_type}_count", current_count + 1)

    def compute(self) -> dict[str, torch.tensor]:
        return {
            f"{self.dataset}_{p}_bias": getattr(self, f"{p}_bias")
            / getattr(self, f"{p}_count")
            for p in self.perturbation_types
        }
