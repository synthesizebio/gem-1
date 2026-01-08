import numpy as np
import torch
import torchmetrics


def pearson_from_aggregated_moments(n, x, x2, y, y2, xy):
    x_mean = x / n
    x_var = x2 / n - x_mean**2
    y_mean = y / n
    y_var = y2 / n - y_mean**2
    r = (xy / n - x_mean * y_mean) / (x_var * y_var) ** 0.5
    r = torch.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)
    return (r * x_var).sum() / (x_var.sum() + 1e-12)


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
