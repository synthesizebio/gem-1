from ... import constants
import torch
import torch.nn as nn
import torch.nn.functional as F
from text_to_rna.gene_order import unalign_gene_expression
from text_to_rna.model.autoencoder.unified_vae import (
    UnifiedVAE,
    classifier_probs_to_dicts,
)
import torch.distributions as dist
from text_to_rna.data import GROUPED_COLUMNS
from text_to_rna.scalers import log_cpm
from cytoolz import merge
import numpy as np


class UnifiedVAENB(UnifiedVAE):
    def __init__(self, config):
        config["use_output_relu"] = False
        UnifiedVAE.__init__(self, config)

    def init_variance(self):
        self.log_theta = nn.ParameterDict(
            {
                modality: nn.Parameter(
                    torch.zeros(
                        constants.MODALITIES[modality]["n_features"]
                        if self.hparams["gene_level_variance"]
                        else 1
                    ),
                    requires_grad=True,
                )
                for modality in self.hparams["modalities"]
            }
        )

    def predictive_mean(
        self,
        total_counts: torch.Tensor,
        z: torch.Tensor,
        modality: str,
        batch_data: dict = None,
    ):
        z_decoded = self.decoder(z)
        logits = unalign_gene_expression(z_decoded, modality)
        if batch_data is not None:
            logits = self._apply_perturbation_adjustment(logits, batch_data, modality)
        return F.softmax(logits, dim=1) * total_counts

    def log_likelihood(
        self,
        x: torch.Tensor,
        z: torch.Tensor,
        modality: str,
        batch_data: dict = None,
        eps=1e-8,
    ) -> torch.Tensor:
        mu = self.predictive_mean(
            batch_data["expression"].sum(dim=1, keepdim=True), z, modality, batch_data
        )
        theta = self.log_theta[modality].exp()
        log_p = torch.log(mu + eps) - torch.log(mu + theta + eps)
        log_1_minus_p = torch.log(theta + eps) - torch.log(mu + theta + eps)
        log_prob = (
            x * log_p
            + theta * log_1_minus_p
            + torch.lgamma(x + theta)
            - torch.lgamma(theta)
            - torch.lgamma(x + 1)
        )
        return log_prob

    def predict_step(
        self,
        batch,
        batch_idx: int = 0,
        dataloader_idx: int = 0,
        sample_decoder: bool = False,
        conditioning=("technical", "biological"),
        total_count: int = 10_000_000,
        fixed_total_count: bool = False,
    ):
        pz = self.prior(batch["metadata_encoding"])
        z = {k: pz[k].sample() for k in GROUPED_COLUMNS.keys()}

        if batch.get("reference_expression") is not None:
            reference = log_cpm(batch["reference_expression"].float())
            qz = self.variational_family(reference, batch["modality"])
            for c in conditioning:
                z[c] = qz[c].sample()

        z_concat = torch.concat([z[k] for k in GROUPED_COLUMNS.keys()], dim=1)

        if batch.get("expression") is not None and not fixed_total_count:
            total_count_tensor = batch["expression"].sum(dim=1, keepdim=True)
        else:
            bs = z_concat.shape[0]
            total_count_tensor = torch.full(
                (bs, 1), float(total_count), device=self.device
            )

        mu = self.predictive_mean(
            total_count_tensor, z_concat, batch["modality"], batch_data=batch
        )

        if sample_decoder:
            theta = self.log_theta[batch["modality"]].exp()
            probs = theta / (theta + mu + 1e-8)
            probs = torch.clamp(probs, 1e-8, 1 - 1e-8)
            px = dist.NegativeBinomial(total_count=theta, probs=probs)
            synthetic_cts = px.sample()
        else:
            synthetic_cts = mu.round()

        out_batch = batch["metadata"].copy()
        out_batch["counts_pred"] = synthetic_cts

        for k, v in z.items():
            out_batch[f"{k}_latent"] = v

        if batch.get("reference_expression") is not None:
            if set(conditioning) == {"technical", "biological"}:
                reference = log_cpm(batch["reference_expression"].float())
                out_batch["de_pred"] = log_cpm(synthetic_cts) - reference
            out_batch["counts_ref"] = synthetic_cts
            out_batch["experiment_accession_ref"] = batch["reference_metadata"].get(
                "experiment_accession", ""
            )

        return out_batch

    def predict_vae(self, batch_data: dict, sample_decoder: bool = False):
        expression_processed = log_cpm(batch_data["expression"])
        qz = self.variational_family(expression_processed, batch_data["modality"])
        latents = {f"{k}_latent": v_dist.sample() for k, v_dist in qz.items()}
        z_posterior = torch.concat(
            [qz[k].rsample() for k in GROUPED_COLUMNS.keys()], dim=1
        )
        total_counts = batch_data["expression"].sum(dim=1, keepdim=True)
        mu = self.predictive_mean(
            total_counts, z_posterior, batch_data["modality"], batch_data=batch_data
        )
        counts_pred = (
            mu.round()
            if not sample_decoder
            else dist.NegativeBinomial(
                total_count=self.log_theta[batch_data["modality"]].exp(),
                logits=mu.log() - self.log_theta[batch_data["modality"]],
            ).sample()
        )
        classifier_probs = self.classifier_probs(qz)
        classifier_dict = classifier_probs_to_dicts(
            keys=self.classifier_keys, probs=classifier_probs
        )
        out_batch = merge(
            batch_data["metadata"],
            classifier_dict,
            latents,
            {"counts": batch_data["expression"]},
            {"counts_pred": counts_pred},
        )
        return out_batch
