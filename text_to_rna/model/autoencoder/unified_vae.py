import torch
import torch.distributions as dist
import torch.nn as nn
from cytoolz import compose_left, curry, pipe, valmap, merge
from cytoolz.curried import map
from omegaconf import OmegaConf, DictConfig
import numpy as np
from ... import constants
from text_to_rna.data import GROUPED_COLUMNS, encode_multilabel
from text_to_rna.gene_order import align_gene_expression, unalign_gene_expression
from text_to_rna.model.base_model import SynthesizeBioModel
from text_to_rna.modules.pff import PFF
from text_to_rna.scalers import log_cpm, log_cpm_inverse
from text_to_rna.vocab import EncodeMetadata
from typing import Union, Any


@torch.no_grad()
def classifier_probs_to_dicts(
    keys: dict[str, list[str]], probs: dict[str, torch.Tensor]
) -> dict[str, Any]:
    """
    Converts raw classifier probabilities (batched tensors) into a more structured
    dictionary format for easier downstream use and storage.

    For each classifier key (e.g., 'cell_type', 'disease'):
    - It determines the predicted class label (string) by taking the argmax of probabilities.
    - It creates a dictionary mapping each possible class label (string) to its probability.

    Args:
        keys: A dictionary mapping classifier names (e.g., "cell_type") to a list of
              possible string labels for that classifier.
        probs: A dictionary mapping classifier names to their corresponding probability
               tensors (shape: [batch_size, num_classes]).
    """
    preds = {}
    for key in probs:
        # Move tensor to CPU before converting to NumPy array for np.argmax
        cpu_probs_for_key = probs[key].cpu().numpy()
        preds[f"{key}_pred"] = [
            keys[key][i] for i in np.argmax(cpu_probs_for_key, axis=1)
        ]

        # Also ensure probabilities for the dict are on CPU
        preds[f"{key}_prob"] = pipe(
            cpu_probs_for_key, map(lambda x: dict(zip(keys[key], x))), list
        )
    return preds


class UnifiedVAE(SynthesizeBioModel):
    """
    Unified Variational Autoencoder (VAE) for gene expression data.

    This model aims to learn disentangled latent representations for different
    sources of variation in gene expression:
    1. Technical variation (e.g., batch effects, library preparation).
    2. Biological variation (e.g., cell type, tissue of origin).
    3. Perturbation effects (e.g., drug treatments, genetic modifications).

    Disentanglement is encouraged through:
    - A metadata-informed prior network that predicts latent distributions from input metadata.
    - Classifiers that predict metadata from their corresponding latent spaces.
    - Adversarial classifiers that penalize a latent space if it encodes information
      that should primarily reside in a different latent space.
    """

    def __init__(self, config):
        assert config["n_pff"] > 0, "Number of pffs need to be greater than 0"

        self.metadata_encoder = EncodeMetadata(
            modalities=config["modalities"],
            min_count=config["min_count_covariates"],
            vectorize=config["vectorize"],
        )
        # Determine input dimensions for metadata predictors based on encoded metadata
        # and set up hidden dimensions for each latent space.
        config["input_dims"] = self.metadata_encoder.dimensions
        config["hidden_dims"] = {
            key: config[f"{key}_hidden_dim"] for key in config["input_dims"].keys()
        }
        config["ae_dim"] = sum(config["hidden_dims"].values())

        # Stores the mapping from perturbation type name (str) to its integer index (int)
        # as defined in the configuration. This is crucial for interpreting one-hot encoded
        # perturbation labels and applying specific adjustments.
        self.perturbation_type_mapping_from_config = None  # Default to None

        # Safely access nested config values
        target_vals_config = config.get("target_vals")
        if isinstance(
            target_vals_config, (dict, DictConfig)
        ):  # Check if target_vals is dict-like
            perturbation_config = target_vals_config.get("perturbation")
            if isinstance(
                perturbation_config, (dict, DictConfig)
            ):  # Check if perturbation is dict-like
                pert_type_config_value = perturbation_config.get("perturbation_type")
                if pert_type_config_value is not None:
                    if OmegaConf.is_dict(pert_type_config_value) or isinstance(
                        pert_type_config_value, dict
                    ):
                        # Config provides a name-to-index dictionary
                        self.perturbation_type_mapping_from_config = (
                            OmegaConf.to_container(pert_type_config_value, resolve=True)
                        )
                        # Ensure values are integers if it's a name-to-index map
                        if not all(
                            isinstance(v, int)
                            for v in self.perturbation_type_mapping_from_config.values()
                        ):
                            print(
                                f"CRITICAL CONFIGURATION WARNING: "
                                f"'target_vals.perturbation.perturbation_type' is a dictionary, "
                                f"but not all values are integers. Found: {self.perturbation_type_mapping_from_config}. "
                                f"Perturbation adjustments may fail or be incorrect."
                            )
                            self.perturbation_type_mapping_from_config = (
                                None  # Invalidate if malformed
                            )
                    else:  # Not a dictionary
                        print(
                            f"CRITICAL CONFIGURATION WARNING: "
                            f"Expected 'target_vals.perturbation.perturbation_type' to be a dictionary "
                            f"(mapping perturbation type name to integer index), "
                            f"but found type '{type(pert_type_config_value)}'. "
                            f"Value: {pert_type_config_value}. "
                            f"Perturbation-specific adjustments will likely not work as intended."
                        )

        # Define the set of genetic perturbation types for targeted adjustment
        self.genetic_perturbation_types = {
            "crispr",
            "shrna",
            "sirna",
            "overexpression",
        }
        SynthesizeBioModel.__init__(self, config)
        self.init_variance()
        self.build_model()
        classifier_keys = merge(self.hparams["target_vals"].values())
        self.classifier_keys = valmap(lambda v: list(v.keys()), classifier_keys)

    def init_variance(self):
        """
        Initializes parameters for the observation noise variance in the VAE's likelihood.
        This can be gene-level (a separate variance parameter for each gene) or
        global (a single variance parameter for all genes within a modality).
        Separate variances are learned for the reconstruction likelihood (log_variance_ae)
        and the prior predictive likelihood (log_variance_pred).
        """
        if self.hparams["gene_level_variance"]:
            self.log_variance_ae = nn.ParameterDict(
                {
                    modality: nn.Parameter(
                        torch.zeros(constants.MODALITIES[modality]["n_features"]),
                        requires_grad=True,
                    )
                    for modality in self.hparams["modalities"]
                }
            )

            self.log_variance_pred = nn.ParameterDict(
                {
                    modality: nn.Parameter(
                        torch.zeros(constants.MODALITIES[modality]["n_features"]),
                        requires_grad=True,
                    )
                    for modality in self.hparams["modalities"]
                }
            )
        else:
            self.log_variance_ae = nn.ParameterDict(
                {
                    modality: nn.Parameter(
                        torch.zeros(1),
                        requires_grad=True,
                    )
                    for modality in self.hparams["modalities"]
                }
            )

            self.log_variance_pred = nn.ParameterDict(
                {
                    modality: nn.Parameter(
                        torch.zeros(1),
                        requires_grad=True,
                    )
                    for modality in self.hparams["modalities"]
                }
            )

    def build_model(self):
        """
        Constructs the main components of the UnifiedVAE model:
        - Metadata predictor networks for each latent space (e.g., technical_predict).
        - The VAE encoder and decoder.
        - Input normalization layers.
        - Primary metadata classifiers and adversarial classifiers.
        """
        for key, input_dim in self.hparams["input_dims"].items():
            hidden_dim = self.hparams["hidden_dims"][key] * 2

            setattr(
                self,
                f"{key}_predict",
                nn.Sequential(
                    # This network predicts the parameters (mean, log_variance) of the prior distribution
                    # for a specific latent space (e.g., 'technical', 'biological', 'perturbation')
                    # based on the input metadata corresponding to that latent space.
                    nn.Dropout(self.hparams[f"{key}_predict_input_dropout"]),
                    nn.Linear(input_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    *[
                        PFF(
                            dim=hidden_dim,
                            expansion_factor=self.hparams["expansion_factor"],
                            activation=self.hparams["activation"],
                            norm=nn.LayerNorm,
                            dropout=self.hparams["pff_dropout"],
                            stochastic_depth=self.hparams["stochastic_depth"],
                        )
                        for _ in range(self.hparams["n_pff"])
                    ],
                    nn.LayerNorm(hidden_dim),
                    nn.Dropout(self.hparams["predict_dropout"]),
                    nn.Linear(hidden_dim, hidden_dim),
                ),
            )

        dim_hidden_ae = self.hparams["dim_hidden_ae"]
        ae_dim = self.hparams["ae_dim"]

        self.encoder = nn.Sequential(
            # The VAE encoder maps gene expression (plus modality indicator)
            # to the parameters (mean, log_variance) of the approximate posterior
            # distribution q(z|x) over the combined latent space.
            nn.Dropout(self.hparams["encoder_input_dropout"]),
            nn.Linear(constants.TOTAL_GENES, dim_hidden_ae),
            nn.LayerNorm(dim_hidden_ae),
            *[
                PFF(
                    dim=dim_hidden_ae,
                    expansion_factor=self.hparams["expansion_factor"],
                    activation=self.hparams["activation"],
                    norm=nn.LayerNorm,
                    dropout=self.hparams["pff_dropout"],
                    stochastic_depth=self.hparams["stochastic_depth"],
                )
                for _ in range(self.hparams["n_pff"])
            ],
            nn.LayerNorm(dim_hidden_ae),
            nn.Dropout(self.hparams["encoder_dropout"]),
            nn.Linear(dim_hidden_ae, ae_dim * 2),
        )

        self.decoder = nn.Sequential(
            # The VAE decoder maps a sample from the latent space (combined latents
            # from technical, biological, perturbation spaces, plus modality indicator)
            # back to the gene expression space, predicting the mean of the reconstruction.
            nn.Dropout(self.hparams["latent_dropout"]),
            nn.Linear(ae_dim, dim_hidden_ae),
            nn.LayerNorm(dim_hidden_ae),
            *[
                PFF(
                    dim=dim_hidden_ae,
                    expansion_factor=self.hparams["expansion_factor"],
                    activation=self.hparams["activation"],
                    norm=nn.LayerNorm,
                    dropout=self.hparams["pff_dropout"],
                    stochastic_depth=self.hparams["stochastic_depth"],
                )
                for _ in range(self.hparams["n_pff"])
            ],
            nn.LayerNorm(dim_hidden_ae),
            nn.Dropout(self.hparams["decoder_dropout"]),
            nn.Linear(
                dim_hidden_ae,
                constants.TOTAL_GENES,
            ),
            nn.ReLU() if self.hparams.get("use_output_relu", True) else nn.Identity(),
        )

        self.input_norm = nn.ModuleDict(
            # Normalization layers applied to the input gene expression
            # for each modality before feeding into the encoder.
            {
                modality: nn.Sequential(
                    nn.BatchNorm1d(constants.MODALITIES[modality]["n_features"], eps=1),
                    nn.LayerNorm(
                        constants.MODALITIES[modality]["n_features"],
                        eps=1,
                    ),
                    nn.ReLU(),
                )
                for modality in self.hparams["modalities"]
            }
        )

        self.classifiers = nn.ModuleDict()
        # Primary classifiers: These predict metadata labels from their
        # corresponding "intended" latent spaces.
        # e.g., predict 'cell_type' (biological metadata) from the 'biological' latent.
        for group, keys in self.hparams["target_vals"].items():
            self.classifiers[group] = nn.ModuleDict(
                {  # Exclude 'study' from primary classifiers
                    key: nn.Sequential(
                        nn.Dropout(self.hparams["classifier_dropout"]),
                        nn.Linear(self.hparams["hidden_dims"][group], len(values)),
                    )
                    for key, values in keys.items()
                    if key not in {"study", "subject_identifier"}
                }
            )

        # Perturbation-specific gene expression adjustment models
        self.perturbation_adjustment_models = nn.ModuleDict()
        # Create a linear model for each combination of modality and genetic perturbation type.
        # These models will learn to adjust the predicted expression of a perturbed gene.
        if hasattr(self, "genetic_perturbation_types"):
            for modality_name in self.hparams["modalities"]:
                for pert_type_name in self.genetic_perturbation_types:
                    model_key = f"{modality_name}_{pert_type_name}"
                    self.perturbation_adjustment_models[model_key] = nn.Linear(1, 1)

    def collate(self, split: str) -> callable:
        return compose_left(
            self.metadata_encoder, curry(encode_multilabel, self.hparams["target_vals"])
        )

    def inject_modality(self, z: torch.Tensor, modality: str) -> torch.Tensor:
        """
        Concatenates a one-hot encoding of the modality to a latent tensor.
        This informs downstream networks (like the decoder) about the data modality.
        """
        modality_encoding = torch.zeros(
            (z.shape[0], len(self.hparams["modalities"])), device=z.device
        )
        modality_encoding[:, self.hparams["modalities"].index(modality)] = 1
        return torch.concat([z, modality_encoding], dim=1)

    def prior(self, metadata_encoding: dict) -> dict[str, dist.Normal]:
        """
        Computes the prior distribution p(z|metadata) for each latent space.
        Each latent space (technical, biological, perturbation) has its own
        predictor network that maps corresponding metadata to its prior parameters.
        """
        priors = {}
        for key, features in metadata_encoding.items():
            z_params = getattr(self, f"{key}_predict")(features.to_dense().float())
            mean, log_variance = torch.split(z_params, z_params.shape[1] // 2, dim=1)
            priors[key] = dist.Normal(mean, log_variance.exp().pow(0.5))
        return priors

    def variational_family(
        self, expression: torch.Tensor, modality: str
    ) -> dict[str, dist.Normal]:
        """
        Computes the approximate posterior distribution q(z|expression, modality)
        using the VAE encoder. The combined latent space from the encoder is then
        split into individual latent spaces (technical, biological, perturbation).
        """
        x = self.input_norm[modality](expression)
        x = align_gene_expression(x, modality)
        z = self.encoder(x)
        mean, log_variance = torch.split(z, z.shape[1] // 2, dim=1)

        i = 0
        q = {}
        for key, dim in self.hparams["hidden_dims"].items():
            q[key] = dist.Normal(
                mean[:, i : i + dim], log_variance[:, i : i + dim].exp().pow(0.5)
            )
            i += dim

        return q

    def _apply_perturbation_adjustment(
        self, mean_expression: torch.Tensor, batch_data: dict, modality: str
    ) -> torch.Tensor:
        """
        Adjusts the input expression tensor (e.g., mean_expression or logits)
        using vectorized operations.

        Args:
            mean_expression: Tensor of shape [batch_size, num_genes_in_modality],
                             representing the decoder's output mean.
            batch_data: The batch data dictionary. Expected to contain:
                        - batch_data["labels"]["perturbation"]["perturbation_type"]:
                          One-hot encoded tensor of shape [batch_size, num_perturbation_types].
                        - batch_data["modality_specific_perturbed_gene_idx"]:
                          Tensor of shape [batch_size] with integer indices for the
                          perturbed gene in the unaligned, modality-specific gene space.
                          Use -1 for samples without a perturbed gene or if unknown.
            modality: The current data modality.

        Returns:
            Adjusted expression tensor. Note: If the base decoder uses ReLU (use_output_relu=True),
            this method also applies ReLU to the adjusted output. Subclasses might override this behavior.
        """
        if (
            not hasattr(self, "perturbation_adjustment_models")
            or not self.perturbation_adjustment_models
        ):
            return mean_expression

        if (
            "labels" not in batch_data
            or "perturbation" not in batch_data["labels"]
            or "perturbation_type" not in batch_data["labels"]["perturbation"]
            or batch_data["labels"]["perturbation"]["perturbation_type"] is None
        ):
            return mean_expression

        pert_type_labels_one_hot = batch_data["labels"]["perturbation"][
            "perturbation_type"
        ].to_dense()
        perturbed_gene_indices = batch_data.get("modality_specific_perturbed_gene_idx")

        if perturbed_gene_indices is None:
            return mean_expression

        adjusted_mean_expression = mean_expression.clone()
        num_genes = mean_expression.shape[1]

        # Mask for samples with a valid (non-sentinel and in-bounds) perturbed gene index
        valid_gene_idx_mask = (
            (perturbed_gene_indices != -1)
            & (perturbed_gene_indices >= 0)
            & (perturbed_gene_indices < num_genes)
        )

        if not torch.any(valid_gene_idx_mask):
            return mean_expression  # No samples with valid perturbed gene indices to adjust

        # Determine the active perturbation type index for each sample
        active_pert_type_indices = torch.argmax(pert_type_labels_one_hot, dim=1)

        # Iterate over the specific genetic perturbation types for which we might apply an adjustment
        for pert_name_to_adjust in self.genetic_perturbation_types:
            model_key = f"{modality}_{pert_name_to_adjust}"
            if model_key not in self.perturbation_adjustment_models:
                continue  # No adjustment model for this modality-perturbation_type combination

            original_config_idx = None
            # Use lowercase for consistent lookup, assuming keys in the config dict are also consistently cased (e.g., lowercase).
            pert_name_lower = pert_name_to_adjust.lower()

            if isinstance(self.perturbation_type_mapping_from_config, dict):
                original_config_idx = self.perturbation_type_mapping_from_config.get(
                    pert_name_lower
                )

            if original_config_idx is None:
                # This pert_name_to_adjust (from self.genetic_perturbation_types)
                # is not found in the config's perturbation_type mapping dictionary,
                # or self.perturbation_type_mapping_from_config is None/malformed.
                # A warning for such mismatches could be logged once during model initialization.
                continue  # Skip if no valid index found for this perturbation type

            # Create a mask for samples in the batch that:
            # 1. Match the current pert_name_to_adjust (via its original_config_idx in the full list)
            # 2. Have a valid perturbed_gene_index
            mask_for_current_pert_model = (
                active_pert_type_indices == original_config_idx
            ) & valid_gene_idx_mask

            if torch.any(mask_for_current_pert_model):
                # Get the specific gene indices for these relevant samples
                gene_indices_to_adjust = perturbed_gene_indices[
                    mask_for_current_pert_model
                ]

                # Extract the VAE's predicted values for these specific genes
                # mean_expression is [batch_size, num_genes]
                # mask_for_current_pert_model is [batch_size] (boolean)
                # gene_indices_to_adjust is [N_relevant_samples] (integer indices)
                # This advanced indexing selects elements from rows matching the mask,
                # at column positions specified by gene_indices_to_adjust.
                original_values_for_perturbed_genes = mean_expression[
                    mask_for_current_pert_model, gene_indices_to_adjust
                ]

                # Reshape for the nn.Linear(1, 1) model: [N_relevant_samples, 1]
                adjustment_input = original_values_for_perturbed_genes.unsqueeze(1)

                # Apply the specific linear model for this perturbation type
                adjustment_deltas = self.perturbation_adjustment_models[model_key](
                    adjustment_input
                )
                adjustment_deltas = adjustment_deltas.squeeze(
                    1
                )  # Back to [N_relevant_samples]

                # Add the adjustments to the corresponding entries in adjusted_mean_expression
                # This uses the same advanced indexing pattern.
                adjusted_mean_expression[
                    mask_for_current_pert_model, gene_indices_to_adjust
                ] = original_values_for_perturbed_genes + adjustment_deltas

        # If the main decoder uses ReLU, ensure the adjusted expression also respects this.
        if self.hparams.get("use_output_relu", True):
            return torch.relu(adjusted_mean_expression)
        else:
            return adjusted_mean_expression

    def likelihood(
        self, z: torch.Tensor, modality: str, is_pred=False, batch_data: dict = None
    ) -> dist.Normal:
        """
        Computes the likelihood p(expression|z, modality) using the VAE decoder.
        This represents the probability of observing the gene expression given a
        latent sample and the modality.
        """
        z_decoded = self.decoder(z)
        mean = unalign_gene_expression(z_decoded, modality)

        if (
            batch_data is not None
        ):  # Apply perturbation adjustment if batch_data is available
            mean = self._apply_perturbation_adjustment(mean, batch_data, modality)

        log_variance = (
            self.log_variance_pred[modality]
            if is_pred
            else self.log_variance_ae[modality]
        )

        return dist.Normal(mean, log_variance.exp().pow(0.5))

    def predict_step(
        self,
        batch,
        batch_idx: int = 0,
        dataloader_idx: int = 0,
        sample_decoder: bool = False,
        conditioning=("technical", "biological"),
        total_count: int = 10_000_000,
        fixed_total_count: bool = False,
        deterministic_latents: bool = False,
    ):
        """
        Generates synthetic gene expression data based on input metadata.
        It samples from the prior p(z|metadata). If reference_expression is provided,
        it can "anchor" the biological and technical latents to those observed in the reference.
        """

        pz = self.prior(batch["metadata_encoding"])
        z = {
            k: (pz[k].mean if deterministic_latents else pz[k].sample())
            for k in GROUPED_COLUMNS.keys()
        }

        if batch.get("reference_expression") is not None:
            reference = log_cpm(batch["reference_expression"].float())
            qz = self.variational_family(reference, batch["modality"])
            for c in conditioning:
                z[c] = qz[c].mean if deterministic_latents else qz[c].sample()

        z_concat = torch.concat([z[k] for k in GROUPED_COLUMNS.keys()], dim=1)

        px = self.likelihood(
            z_concat, batch["modality"], is_pred=True, batch_data=batch
        )

        if batch.get("expression") is not None and not fixed_total_count:
            total_count = batch["expression"].sum(dim=1, keepdim=True)

        synthetic_cts = log_cpm_inverse(
            px.sample() if sample_decoder else px.loc, total_cts=total_count
        )

        out_batch = batch["metadata"].copy()
        out_batch["counts_pred"] = synthetic_cts

        for k, v in z.items():
            out_batch[f"{k}_latent"] = v

        if batch.get("reference_expression") is not None:
            if set(conditioning) == {"technical", "biological"}:
                # if we are generating a perturbation, calculate de
                out_batch["de_pred"] = log_cpm(synthetic_cts) - reference
            out_batch["counts_ref"] = reference
            out_batch["experiment_accession_ref"] = batch["reference_metadata"].get(
                "experiment_accession", ""
            )

        return out_batch

    def predict_vae(
        self,
        batch_data: dict,
        sample_decoder: bool = False,
        deterministic_latents: bool = False,
    ):
        """
        Computes all components related to the VAE (encoder-decoder) pathway:
        - Reconstructed gene expression.
        - Probabilities of metadata categories.
        """
        qz = self.encode_expression(
            batch_data["expression"],
            batch_data["modality"],
            deterministic_latents=deterministic_latents,
        )

        for k, v in qz.items():
            batch_data[f"{k}_latent"] = v

        z_posterior = torch.concat([qz[k] for k in GROUPED_COLUMNS.keys()], dim=1)

        posterior_predictive = self.likelihood(
            z_posterior, batch_data["modality"], is_pred=False, batch_data=batch_data
        )
        total_counts = batch_data["expression"].sum(dim=1, keepdim=True)
        classifier_probs = self.classifier_probs(qz)
        classifier_dict = classifier_probs_to_dicts(
            keys=self.classifier_keys, probs=classifier_probs
        )

        counts_pred = log_cpm_inverse(
            posterior_predictive.sample()
            if sample_decoder
            else posterior_predictive.loc,
            total_cts=total_counts,
        )

        out_batch = merge(
            batch_data["metadata"],
            classifier_dict,
            {"counts": batch_data["expression"]},
            {"counts_pred": counts_pred},
        )
        return out_batch

    def encode_expression(
        self, expression, modality, deterministic_latents: bool = False
    ):
        """
        Encodes input gene expression into the disentangled latent spaces
        (technical, biological, perturbation) using the VAE encoder.
        """
        qz = self.variational_family(log_cpm(expression), modality)
        return {
            k: (qz[k].mean if deterministic_latents else qz[k].rsample())
            for k in GROUPED_COLUMNS.keys()
        }

    def classifier_probs(
        self, z: dict[str, Union[torch.distributions.Distribution, torch.Tensor]]
    ):
        """
        Given samples from the latent spaces, computes the probability distributions
        over metadata categories using the primary classifiers.
        """
        probs = {}
        for group in ["biological", "perturbation"]:
            z_group = (
                z[group] if isinstance(z[group], torch.Tensor) else z[group].sample()
            )
            for key in self.classifiers[group].keys():
                probs[key] = torch.nn.functional.softmax(
                    self.classifiers[group][key](z_group), dim=1
                )
        return probs
