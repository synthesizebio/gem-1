<p align="center">
  <img src="assets/logo.png" alt="Synthesize Bio Logo" width="400"/>
</p>

# Synthesize Bio GEM‑1 Preprint Toolkit

This repository focuses on the modeling and inference code that powers Synthesize Bio's GEM‑1 model preprint. It provides the core implementation for loading GEM‑1 models and running inference in downstream workflows.

The core Python package (`text_to_rna`) implements model loading and inference, while `inference_scripts/` provides the command-line entry points for predictions.

---

## 🧭 Table of Contents

* [Getting Started](#-getting-started)
* [Repository Structure](#-repository-structure)
* [The GEM-1 Models](#-the-gem-1-models)
* [Using the GEM-1 Model (Inference)](#-using-the-gem-1-model-inference)
* [Citation](#-citation)
* [Troubleshooting](#-troubleshooting)

---

## 🚀 Getting Started

### Prerequisites

* Python 3.10+ with `pip` (GPU-enabled PyTorch optional but recommended for speed)
* (Optional) Jupyter Lab / Notebook for `.ipynb` workflows

### Environment setup

```bash
# Clone the repository
git clone https://github.com/synthesizebio/synthesizebio-gem1-toolkit.git
cd synthesizebio-gem1-toolkit

# Create conda environment
conda create -n synthesize_bio_preprint_2025 python=3.11 -y
conda activate synthesize_bio_preprint_2025
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

**or using uv**

```bash
# Clone the repository
git clone https://github.com/synthesizebio/synthesizebio-gem1-toolkit.git
cd synthesizebio-gem1-toolkit

# Create environment
uv venv --python 3.11 .venv
source .venv/bin/activate            # macOS/Linux
# .venv\Scripts\activate             # Windows PowerShell/CMD
uv pip install -r requirements.txt
```

---

## 📂 Repository Structure

* **text_to_rna/** – Python package implementing GEM‑1 data utilities, pretrained weights, scalers, and Lightning model wrappers.
* **inference_scripts/** – Command-line entry point for running GEM‑1 predictions (`inference.py`).
* **logs/** – Placeholder for experiment or script logs.
* **requirements.txt** – Project-wide Python dependencies baseline.

---

## 🧬 The GEM-1 Models

GEM-1 ships as a pair of Lightning modules plus an inference harness. `inference_scripts/inference.py:load_model` automatically picks the correct class for a modality and expects checkpoints under `data/model_files/<modality>/{config.yaml,last.ckpt}`.

### Model Variants and Checkpoints

* **Bulk** (`text_to_rna/model/autoencoder/unified_vae.py`) – `UnifiedVAE` disentangles technical, biological, and perturbation latents by conditioning its priors on metadata (`GROUPED_COLUMNS`), training metadata classifiers on each latent block, and decoding multinomial gene-expression distributions. It is the default whenever `--modality bulk`.

* **Single-cell** (`text_to_rna/model/autoencoder/unified_vae_cce.py`) – `UnifiedVAECCE` extends the negative-binomial variant (`UnifiedVAENB`) to work with compositional cell embeddings. It consumes raw single-cell counts, models per-cell library sizes, and shares the same prediction API.

---

## ⚙️ Using the GEM-1 Model (Inference)

You can either run the CLI helpers or import the underlying functions in your own pipelines.

### Inference entry point (inference_scripts/inference.py)

The script streams Parquet rows, pads metadata via `text_to_rna.data.expand_dict`, batches them with `data.collate`, and routes each batch through the selected model. Outputs are written as Hive-partitioned Parquet datasets (`study=<id>`). The CLI exposes the following high-level tasks:

| Task (`--task`) | Backing function(s) | Use it when | Strategy notes |
|-----------------|---------------------|-------------|----------------|
| `predict_expression` | `parquet_to_parquet` → `predict_stream` (independent) or `predict_common_technical` (shared reference) | You only have metadata and want GEM‑1 to forecast expression counts. | Add `--use_common_technical` to synthesize one reference per technical group, attach it as `counts_ref`, and then condition predictions on the shared technical latent; omit it (or pass `--no-use_common_technical`) to treat each row independently. |
| `ref_predict_perturbation` | `ref_parquet_to_parquet` → `predict_perturbation_ref_study` | You observed matched controls and perturbations in the same study and want GEM‑1 to estimate perturbation deltas. | The helper pairs each perturbation with its control via `to_ref_conditioned_pert*`, keeps control rows (unless `--classification_ready_output` is set), and predicts conditioned on both technical and biological latents. Outputs include `counts_pred`, the original controls (with `counts` renamed to `counts_pred`), and optional `de_pred`. |
| `predict_perturbation` | `predict_perturbation_fully_synthetic` → `substitute_controls` + `predict_perturbation_ref_study` | You lack measured controls but still want counterfactual perturbation predictions. | Within every technical group the script first uses `predict_common_technical` to synthesize control counts, replaces the real control rows with those predictions, and then runs the same reference-conditioned pipeline as above. Ideal for "what-if" dosing studies. |
| `classify_samples` | `classify_parquet` → `classify_samples` | You have expression matrices and want the model's latent classifiers (cell type, tissue, disease, etc.). | GEM‑1 encodes expression, samples latents via `model.predict_vae`, emits reconstructed counts, and returns one-hot probability dictionaries for each classifier (`*_pred`, `*_prob`). |

Every task trims inputs to `text_to_rna.data.TABLE_COLUMNS`, keeps memory bounded by streaming one study at a time, and writes results with snappy compression for downstream analytics.

### Prediction modes and knobs

* **--mode {mean estimation,sample generation}** – `mean estimation` returns the decoder expectation (rounded counts) while `sample generation` draws from the Multinomial / Negative-Binomial likelihood for stochastic replicates.
* **--replicates N** – Duplicates each metadata row N times before inference so you can form prediction ensembles or Monte Carlo samples.
* **--use_common_technical / --no-use_common_technical** – Only relevant for `predict_expression`; toggles whether each technical group shares a latent "reference" prediction via `predict_common_technical`.
* **--classification_ready_output** – For reference-conditioned perturbation tasks, drop control rows and rename `counts_pred`→`counts` so downstream classifiers can ingest only perturbed predictions.
* **--total_count + --fixed_total_count** – Set or override the target library size. When `fixed_total_count` is false, the model copies totals from observed counts (if available); otherwise it scales every prediction to `total_count`.
* **--batch_size / --output_batch_size** – Tune GPU throughput and how many rows land in each Parquet part file.
* **--strict** – Disable to allow checkpoint loading when keys drift from the saved `config.yaml` (useful for rapid experimentation).

### Input & output checklist

* **Input Parquet dataset** – Either a single file or a Hive-partitioned directory. Must contain `experiment_accession`, a `study` column (or partitions), and the metadata fields listed in `text_to_rna.data.METADATA_COLUMNS`. Provide `counts` for classification or observed-control perturbation tasks; provide `counts_ref` if you precomputed references yourself.
* **Output path** – Target directory for partitioned Parquet shards; the script creates `study=<accession>/part-*.parquet` folders and warns if it is about to append.
* **Modality + checkpoint** – Choose `bulk` (loads `UnifiedVAE`) or `single_cell` (loads `UnifiedVAECCE`).
* **Conditioning metadata** – Populate the group keys (`text_to_rna.data.GROUPED_COLUMNS["technical"]` and `["biological"]`) so the model can align controls and perturbations inside each technical group.
* **Optional references** – Fields such as `counts_ref`, `ref_experiment_accession_hint`, or prior DE tables are automatically propagated into the prediction dictionaries if present.


### Example workflows

```python
from inference_scripts.inference import parquet_to_parquet

parquet_to_parquet(
    modality="bulk",
    input_path="path/to/input_parquet",
    output_path="path/to/output_parquet",
    mode="mean estimation",
    total_count=10_000_000,
    fixed_total_count=False,
    batch_size=128,
    replicates=1,
)
```

Swap in `ref_parquet_to_parquet` when you have observed controls, `predict_perturbation_fully_synthetic` for "no-control" simulations, or `classify_parquet` when you need latent classifier scores. Refer to `inference_scripts/inference.py` for the full set of arguments and defaults, including how to toggle `--classification_ready_output` for perturbation-specific exports.

---

## 📜 Citation

If you use GEM-1 or this toolkit in your research, please cite the preprint:

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

---

## 🔗 Links

* **Preprint**: [bioRxiv 2025.09.08.674753](https://www.biorxiv.org/content/10.1101/2025.09.08.674753v1)
* **Company Website**: [www.synthesize.bio](https://www.synthesize.bio)

---

## 💡 Troubleshooting

* Inference workloads expect a GPU for timely execution, but CPU runs are supported (set `CUDA_VISIBLE_DEVICES=` to force CPU).
* If you encounter `ModuleNotFoundError` for `text_to_rna`, ensure you either run commands from the repository root or explicitly set `PYTHONPATH`.
* Keep large outputs (predictions, metrics, plots) inside the provided folder structure so that downstream workflows can locate them without additional configuration.

Happy modeling!
