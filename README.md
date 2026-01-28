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
