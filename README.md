<p align="center">
  <img src="assets/logo.png" alt="Synthesize Bio Logo" width="400"/>
</p>

# GEM Model

This repository provides a focused implementation of the GEM model as a standard Python package.

## What’s Included

- `src/gem/model.py` — model definition, metadata encoder, synthetic dataset, and a small training demo.

## Quickstart

```bash
python -m pip install -r requirements.txt
python -m gem
```

Installation should take ~1–3 minutes on a typical laptop (longer if PyTorch wheels need to download).

## Mock Training Script

Run the synthetic training demo:

```bash
python -m gem
```

Expected output (per epoch):

```
epoch 01 | loss ... | recon ... | kl ...
```

On CPU, the demo typically finishes in ~30–90 seconds for the default config (longer on older machines). On GPU, it should finish in a few seconds.

Warning: the dataset is random and has no true structure to learn, so reconstruction loss is not expected to converge. The goal is to verify the training loop and data flow, not model quality.

## Notes

- Inputs are converted to log-CPM and normalized before encoding.
- The decoder outputs a Normal distribution in log-CPM space.
- `TrainConfig` in `src/gem/model.py` controls model and dataset sizes.

## Architecture (Pseudocode)

```text
# Inputs
expression: raw counts (batch x genes)
metadata: structured covariates (batch)

# --- Prediction / Generation from metadata (prior) ---
metadata_encoding = EncodeMetadata(metadata)
for group in {technical, biological, perturbation}:
    prior_params[group] = MLP_predict[group](metadata_encoding[group])
    z_prior[group] = sample_or_mean(Normal(prior_params[group]))

if reference_expression provided:
    # optional anchoring to observed data
    ref_q = Encoder(log_cpm(reference_expression))
    z_prior[group in conditioning] = sample_or_mean(ref_q[group])

z_concat = concat(z_prior[technical], z_prior[biological], z_prior[perturbation])
decoded = Decoder(z_concat)
px = Normal(mean=decoded, variance=learned)
counts_pred = log_cpm_inverse(sample_or_mean(px), total_counts)

# --- Reconstruction / VAE path (used in training) ---
x = log_cpm(expression)
encoder_out = Encoder(x)
for group in {technical, biological, perturbation}:
    q_params[group] = split(encoder_out)
    z_post[group] = rsample_or_mean(Normal(q_params[group]))

z_concat = concat(z_post[technical], z_post[biological], z_post[perturbation])
decoded = Decoder(z_concat)
px = Normal(mean=decoded, variance=learned)
counts_recon = log_cpm_inverse(sample_or_mean(px), total_counts)

# --- Auxiliary heads (train-time signals) ---
for group in {biological, perturbation}:
    for label_head in classifiers[group]:
        probs[label_head] = softmax(Classifier[group][label_head](z_post[group]))

# Training uses:
# - reconstruction likelihood from px
# - KL(q(z|x) || p(z|metadata)) per group
# - classifier losses on intended latent groups
```

## Citation

If you use GEM or this toolkit in your research, please cite the preprint:

```bibtex
@article{gem1_2025,
  title={GEM-1: A Foundation Model for Gene Expression Prediction},
  author={Koytiger, Gregory and Walsh, Alice and Bradley, Robert and Leek, Jeffrey and [Additional Authors]},
  journal={bioRxiv},
  year={2025},
  doi={10.1101/2025.09.08.674753},
  url={https://www.biorxiv.org/content/10.1101/2025.09.08.674753v1}
}
```

## Links

- Preprint: https://www.biorxiv.org/content/10.1101/2025.09.08.674753v1
- Company website: https://www.synthesize.bio
