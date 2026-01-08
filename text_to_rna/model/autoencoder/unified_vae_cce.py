import torch
from text_to_rna.gene_order import unalign_gene_expression
from text_to_rna.model.autoencoder.unified_vae_nb import UnifiedVAENB
from text_to_rna.model.autoencoder.unified_vae import classifier_probs_to_dicts
from text_to_rna.scalers import log_cpm
from text_to_rna.data import GROUPED_COLUMNS
import torch.distributions as dist
from cytoolz import merge


class UnifiedVAECCE(UnifiedVAENB):
    def __init__(self, config):
        UnifiedVAENB.__init__(self, config)

    def likelihood(
        self,
        sample: torch.Tensor,
        modality: str,
        total_count: torch.Tensor,
        batch_data: dict = None,
    ) -> torch.Tensor:
        # total_count is unused here, present to conform to parent/grandparent class signatures
        logits = self.decoder(sample)
        unaligned_logits = unalign_gene_expression(logits, modality)
        if batch_data is not None and modality != "czi":
            return self._apply_perturbation_adjustment(
                unaligned_logits, batch_data, modality
            )
        return unaligned_logits

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

        prior_logits = self.likelihood(
            z_concat, batch["modality"], total_count=None, batch_data=batch
        )

        if sample_decoder:
            if batch.get("expression") is not None and not fixed_total_count:
                total_count_tensor = batch["expression"].sum(dim=1)
            else:
                bs = z_concat.shape[0]
                total_count_tensor = torch.full(
                    (bs,), float(total_count), device=self.device
                )
            px = dist.Multinomial(
                total_count=total_count_tensor.int(), logits=prior_logits
            )
            synthetic_cts = px.sample()
        else:
            probs = torch.softmax(prior_logits, dim=1)
            if batch.get("expression") is not None and not fixed_total_count:
                total_count_tensor = batch["expression"].sum(dim=1, keepdim=True)
            else:
                bs = z_concat.shape[0]
                total_count_tensor = torch.full(
                    (bs, 1), float(total_count), device=self.device
                )
            synthetic_cts = (probs * total_count_tensor).round()

        out_batch = batch["metadata"].copy()
        out_batch["counts_pred"] = synthetic_cts

        for k, v in z.items():
            out_batch[f"{k}_latent"] = v

        if (
            batch.get("reference_expression") is not None
            and "reference_metadata" in batch
        ):
            if set(conditioning) == {"technical", "biological"}:
                reference = torch.log1p(batch["reference_expression"].float())
                out_batch["de_pred"] = torch.log1p(synthetic_cts) - reference
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
        posterior_logits = self.likelihood(
            z_posterior, batch_data["modality"], total_count=None, batch_data=batch_data
        )
        total_counts = batch_data["expression"].sum(dim=1, keepdim=True)
        if sample_decoder:
            px = dist.Multinomial(
                total_count=total_counts.squeeze().int(), logits=posterior_logits
            )
            counts_pred = px.sample()
        else:
            probs = torch.softmax(posterior_logits, dim=1)
            counts_pred = (probs * total_counts).round()
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
