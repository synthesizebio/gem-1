"""
GEM model distilled into a single file for easy iteration.

Run:
  python -m gem
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ---------------------------
# PFF
# ---------------------------
class PFF(nn.Module):
    def __init__(self, dim: int, expansion_factor: int = 2, dropout: float = 0.0):
        super().__init__()
        hidden_dim = dim * expansion_factor
        self.ff = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.ff(x)


# ---------------------------
# Metadata encoder
# ---------------------------
class MetadataEncoder(nn.Module):
    def __init__(self, n_technical: int, n_biological: int, n_perturbation: int):
        super().__init__()
        self.n_technical = n_technical
        self.n_biological = n_biological
        self.n_perturbation = n_perturbation

    @property
    def dimensions(self) -> Dict[str, int]:
        return {
            "technical": self.n_technical,
            "biological": self.n_biological,
            "perturbation": self.n_perturbation,
        }

    def forward(self, metadata: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        tech = F.one_hot(metadata["technical"].long(), num_classes=self.n_technical)
        bio = F.one_hot(metadata["biological"].long(), num_classes=self.n_biological)
        pert = F.one_hot(
            metadata["perturbation"].long(), num_classes=self.n_perturbation
        )
        return {
            "technical": tech.float(),
            "biological": bio.float(),
            "perturbation": pert.float(),
        }


def log_cpm(counts: torch.Tensor, total_cts: torch.Tensor | None = None) -> torch.Tensor:
    if total_cts is None:
        total_cts = counts.sum(dim=1, keepdim=True).clamp_min(1.0)
    scaled = counts / total_cts * 1_000_000.0
    return torch.log1p(scaled)


def log_cpm_inverse(
    log_cpm_values: torch.Tensor, total_cts: torch.Tensor
) -> torch.Tensor:
    return torch.expm1(log_cpm_values) * total_cts / 1_000_000.0


# ---------------------------
# GEM model
# ---------------------------
class GEM(nn.Module):
    def __init__(
        self,
        n_genes: int,
        n_technical: int,
        n_biological: int,
        n_perturbation: int,
        tech_dim: int = 8,
        bio_dim: int = 12,
        pert_dim: int = 8,
        hidden_dim: int = 128,
        beta: float = 1.0,
        n_pff: int = 1,
        expansion_factor: int = 2,
        pff_dropout: float = 0.0,
    ):
        super().__init__()
        self.latent_groups: List[str] = ["technical", "biological", "perturbation"]
        self.hidden_dims: Dict[str, int] = {
            "technical": tech_dim,
            "biological": bio_dim,
            "perturbation": pert_dim,
        }
        self.total_latent = sum(self.hidden_dims.values())
        self.beta = beta
        self.n_pff = n_pff
        self.expansion_factor = expansion_factor
        self.pff_dropout = pff_dropout

        self.metadata_encoder = MetadataEncoder(
            n_technical=n_technical,
            n_biological=n_biological,
            n_perturbation=n_perturbation,
        )
        self.input_dims = self.metadata_encoder.dimensions

        self.prior_nets = nn.ModuleDict(
            {
                group: self._mlp(self.input_dims[group], hidden_dim, 2 * dim)
                for group, dim in self.hidden_dims.items()
            }
        )

        self.input_norm = nn.LayerNorm(n_genes, eps=1.0)

        self.encoder = self._mlp(n_genes, hidden_dim, 2 * self.total_latent)
        self.decoder = self._mlp(self.total_latent, hidden_dim, n_genes)
        self.log_variance = nn.Parameter(torch.zeros(n_genes), requires_grad=True)

    def _mlp(self, in_dim: int, hidden_dim: int, out_dim: int) -> nn.Sequential:
        blocks = [
            nn.Linear(in_dim, hidden_dim),
        ]
        blocks.extend(
            [
                PFF(
                    dim=hidden_dim,
                    expansion_factor=self.expansion_factor,
                    dropout=self.pff_dropout,
                )
                for _ in range(self.n_pff)
            ]
        )
        blocks.extend([nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, out_dim)])
        return nn.Sequential(*blocks)

    def _split_stats(self, mean: torch.Tensor, logvar: torch.Tensor):
        stats = {}
        idx = 0
        for group in self.latent_groups:
            dim = self.hidden_dims[group]
            stats[group] = {
                "mean": mean[:, idx : idx + dim],
                "logvar": logvar[:, idx : idx + dim],
            }
            idx += dim
        return stats

    @staticmethod
    def _rsample(mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        eps = torch.randn_like(mean)
        return mean + eps * torch.exp(0.5 * logvar)

    @staticmethod
    def _kl_normal(
        q_mean: torch.Tensor,
        q_logvar: torch.Tensor,
        p_mean: torch.Tensor,
        p_logvar: torch.Tensor,
    ) -> torch.Tensor:
        q_var = torch.exp(q_logvar)
        p_var = torch.exp(p_logvar)
        kl = 0.5 * (
            p_logvar
            - q_logvar
            + (q_var + (q_mean - p_mean) ** 2) / p_var
            - 1.0
        )
        return kl.sum(dim=1).mean()

    def prior(self, metadata_encoding: Dict[str, torch.Tensor]):
        priors = {}
        for group in self.latent_groups:
            params = self.prior_nets[group](metadata_encoding[group])
            mean, logvar = torch.chunk(params, 2, dim=1)
            priors[group] = {"mean": mean, "logvar": logvar}
        return priors

    def encode(self, x: torch.Tensor):
        params = self.encoder(x)
        mean, logvar = torch.chunk(params, 2, dim=1)
        return self._split_stats(mean, logvar)

    def decode(self, z_concat: torch.Tensor):
        return self.decoder(z_concat)

    def forward(self, counts: torch.Tensor, metadata: Dict[str, torch.Tensor]):
        metadata = {k: v.to(counts.device) for k, v in metadata.items()}
        x = log_cpm(counts)
        x = self.input_norm(x)

        q_stats = self.encode(x)
        z = {
            group: self._rsample(q_stats[group]["mean"], q_stats[group]["logvar"])
            for group in self.latent_groups
        }
        z_concat = torch.cat([z[group] for group in self.latent_groups], dim=1)
        x_recon = self.decode(z_concat)

        metadata_encoding = self.metadata_encoder(metadata)
        p_stats = self.prior(metadata_encoding)

        recon_dist = dist.Normal(
            loc=x_recon, scale=torch.exp(0.5 * self.log_variance)
        )
        recon_loss = -recon_dist.log_prob(x).mean()
        kl_loss = 0.0
        for group in self.latent_groups:
            kl_loss = kl_loss + self._kl_normal(
                q_stats[group]["mean"],
                q_stats[group]["logvar"],
                p_stats[group]["mean"],
                p_stats[group]["logvar"],
            )

        loss = recon_loss + self.beta * kl_loss

        return {
            "loss": loss,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
            "x_recon": x_recon,
        }

    @torch.no_grad()
    def generate_from_metadata(
        self,
        metadata: Dict[str, torch.Tensor],
        sample: bool = True,
        total_counts: torch.Tensor | None = None,
    ) -> torch.Tensor:
        device = self.log_variance.device
        metadata = {k: v.to(device) for k, v in metadata.items()}
        metadata_encoding = self.metadata_encoder(metadata)
        p_stats = self.prior(metadata_encoding)
        z = {}
        for group in self.latent_groups:
            if sample:
                z[group] = self._rsample(p_stats[group]["mean"], p_stats[group]["logvar"])
            else:
                z[group] = p_stats[group]["mean"]
        z_concat = torch.cat([z[group] for group in self.latent_groups], dim=1)
        x_mean = self.decode(z_concat)
        recon_dist = dist.Normal(
            loc=x_mean, scale=torch.exp(0.5 * self.log_variance)
        )
        x_sample = recon_dist.sample() if sample else x_mean

        if total_counts is None:
            total_counts = torch.full(
                (x_sample.shape[0], 1),
                1_000_000.0,
                device=x_sample.device,
            )
        else:
            total_counts = total_counts.to(x_sample.device)

        return log_cpm_inverse(x_sample, total_cts=total_counts).round().clamp_min(0.0)

    @torch.no_grad()
    def generate_from_reference(
        self,
        reference_counts: torch.Tensor,
        metadata: Dict[str, torch.Tensor],
        conditioning: Tuple[str, ...] = ("technical", "biological"),
        sample: bool = True,
        total_counts: torch.Tensor | None = None,
    ) -> torch.Tensor:
        device = self.log_variance.device
        reference_counts = reference_counts.to(device)
        metadata = {k: v.to(device) for k, v in metadata.items()}

        x_ref = self.input_norm(log_cpm(reference_counts))
        q_stats = self.encode(x_ref)
        p_stats = self.prior(self.metadata_encoder(metadata))

        z = {}
        for group in self.latent_groups:
            if group in conditioning:
                mean = q_stats[group]["mean"]
                logvar = q_stats[group]["logvar"]
            else:
                mean = p_stats[group]["mean"]
                logvar = p_stats[group]["logvar"]
            z[group] = self._rsample(mean, logvar) if sample else mean

        z_concat = torch.cat([z[group] for group in self.latent_groups], dim=1)
        x_mean = self.decode(z_concat)
        recon_dist = dist.Normal(
            loc=x_mean, scale=torch.exp(0.5 * self.log_variance)
        )
        x_sample = recon_dist.sample() if sample else x_mean

        if total_counts is None:
            total_counts = reference_counts.sum(dim=1, keepdim=True).clamp_min(1.0)
        else:
            total_counts = total_counts.to(device)

        return log_cpm_inverse(x_sample, total_cts=total_counts).round().clamp_min(0.0)


# ---------------------------
# Mock dataset
# ---------------------------
class GeneExpressionDataset(Dataset):
    def __init__(
        self,
        n_samples: int,
        n_genes: int,
        n_technical: int,
        n_biological: int,
        n_perturbation: int,
        tech_dim: int,
        bio_dim: int,
        pert_dim: int,
        seed: int = 7,
    ):
        super().__init__()
        torch.manual_seed(seed)

        self.n_samples = n_samples
        self.n_genes = n_genes

        tech_idx = torch.randint(0, n_technical, (n_samples,))
        bio_idx = torch.randint(0, n_biological, (n_samples,))
        pert_idx = torch.randint(0, n_perturbation, (n_samples,))

        tech_embed = torch.randn(n_technical, tech_dim) * 0.5
        bio_embed = torch.randn(n_biological, bio_dim) * 0.5
        pert_embed = torch.randn(n_perturbation, pert_dim) * 0.5

        z_tech = tech_embed[tech_idx] + 0.1 * torch.randn(n_samples, tech_dim)
        z_bio = bio_embed[bio_idx] + 0.1 * torch.randn(n_samples, bio_dim)
        z_pert = pert_embed[pert_idx] + 0.1 * torch.randn(n_samples, pert_dim)

        z_concat = torch.cat([z_tech, z_bio, z_pert], dim=1)
        w = torch.randn(tech_dim + bio_dim + pert_dim, n_genes) * 0.2
        b = torch.randn(n_genes) * 0.1

        rate = F.softplus(z_concat @ w + b) + 1e-3
        counts = torch.poisson(rate)

        self.counts = counts.float()
        self.metadata = {
            "technical": tech_idx,
            "biological": bio_idx,
            "perturbation": pert_idx,
        }

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        metadata = {
            "technical": self.metadata["technical"][idx],
            "biological": self.metadata["biological"][idx],
            "perturbation": self.metadata["perturbation"][idx],
        }
        return self.counts[idx], metadata


# ---------------------------
# Training demo
# ---------------------------
@dataclass
class TrainConfig:
    n_samples: int = 2048
    n_genes: int = 256
    n_technical: int = 4
    n_biological: int = 6
    n_perturbation: int = 3
    tech_dim: int = 8
    bio_dim: int = 12
    pert_dim: int = 8
    hidden_dim: int = 128
    beta: float = 1.0
    n_pff: int = 1
    expansion_factor: int = 2
    pff_dropout: float = 0.0
    batch_size: int = 64
    epochs: int = 5
    lr: float = 1e-3
    device: str = "cpu"


def train_demo(cfg: TrainConfig):
    dataset = GeneExpressionDataset(
        n_samples=cfg.n_samples,
        n_genes=cfg.n_genes,
        n_technical=cfg.n_technical,
        n_biological=cfg.n_biological,
        n_perturbation=cfg.n_perturbation,
        tech_dim=cfg.tech_dim,
        bio_dim=cfg.bio_dim,
        pert_dim=cfg.pert_dim,
    )

    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

    model = GEM(
        n_genes=cfg.n_genes,
        n_technical=cfg.n_technical,
        n_biological=cfg.n_biological,
        n_perturbation=cfg.n_perturbation,
        tech_dim=cfg.tech_dim,
        bio_dim=cfg.bio_dim,
        pert_dim=cfg.pert_dim,
        hidden_dim=cfg.hidden_dim,
        beta=cfg.beta,
        n_pff=cfg.n_pff,
        expansion_factor=cfg.expansion_factor,
        pff_dropout=cfg.pff_dropout,
    ).to(cfg.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    model.train()
    for epoch in range(1, cfg.epochs + 1):
        total_loss = 0.0
        total_recon = 0.0
        total_kl = 0.0
        for counts, metadata in loader:
            counts = counts.to(cfg.device)
            metadata = {k: v.to(cfg.device) for k, v in metadata.items()}

            out = model(counts, metadata)
            loss = out["loss"]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_recon += out["recon_loss"].item()
            total_kl += out["kl_loss"].item()

        steps = len(loader)
        print(
            f"epoch {epoch:02d} | "
            f"loss {total_loss / steps:.4f} | "
            f"recon {total_recon / steps:.4f} | "
            f"kl {total_kl / steps:.4f}"
        )

    return model


if __name__ == "__main__":
    cfg = TrainConfig()
    train_demo(cfg)
