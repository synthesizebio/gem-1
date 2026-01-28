import torch

from gem.model import GEM


def _make_batch(batch_size=4, n_genes=16, n_technical=3, n_biological=4, n_perturbation=2):
    torch.manual_seed(0)
    counts = torch.poisson(torch.rand(batch_size, n_genes) * 5.0 + 1.0).float()
    metadata = {
        "technical": torch.randint(0, n_technical, (batch_size,)),
        "biological": torch.randint(0, n_biological, (batch_size,)),
        "perturbation": torch.randint(0, n_perturbation, (batch_size,)),
    }
    return counts, metadata


def _make_model(n_genes=16, n_technical=3, n_biological=4, n_perturbation=2):
    return GEM(
        n_genes=n_genes,
        n_technical=n_technical,
        n_biological=n_biological,
        n_perturbation=n_perturbation,
        tech_dim=4,
        bio_dim=5,
        pert_dim=3,
        hidden_dim=32,
        n_pff=1,
    )


def test_forward_outputs():
    n_genes = 16
    model = _make_model(n_genes=n_genes)
    counts, metadata = _make_batch(n_genes=n_genes)

    out = model(counts, metadata)

    assert "loss" in out and "recon_loss" in out and "kl_loss" in out
    assert out["loss"].ndim == 0
    assert out["recon_loss"].ndim == 0
    assert out["kl_loss"].ndim == 0
    assert out["x_recon"].shape == (counts.shape[0], n_genes)
    assert torch.isfinite(out["loss"]).all()


def test_generate_from_metadata_shape_and_device():
    n_genes = 12
    n_technical, n_biological, n_perturbation = 3, 4, 2
    model = _make_model(
        n_genes=n_genes,
        n_technical=n_technical,
        n_biological=n_biological,
        n_perturbation=n_perturbation,
    )

    metadata = {
        "technical": torch.tensor([0, 1, 2, 0]),
        "biological": torch.tensor([1, 2, 3, 0]),
        "perturbation": torch.tensor([0, 1, 0, 1]),
    }
    total_counts = torch.tensor([[1_000_000.0]]).repeat(4, 1)

    counts = model.generate_from_metadata(metadata, sample=False, total_counts=total_counts)

    assert counts.shape == (4, n_genes)
    assert torch.all(counts >= 0)

    if torch.cuda.is_available():
        model = model.cuda()
        counts_gpu = model.generate_from_metadata(metadata, sample=False, total_counts=total_counts)
        assert counts_gpu.is_cuda
        assert counts_gpu.shape == (4, n_genes)


def test_generate_from_reference_shape():
    n_genes = 10
    n_technical, n_biological, n_perturbation = 3, 4, 2
    model = _make_model(
        n_genes=n_genes,
        n_technical=n_technical,
        n_biological=n_biological,
        n_perturbation=n_perturbation,
    )

    counts, metadata = _make_batch(
        batch_size=5,
        n_genes=n_genes,
        n_technical=n_technical,
        n_biological=n_biological,
        n_perturbation=n_perturbation,
    )

    out = model.generate_from_reference(
        reference_counts=counts,
        metadata=metadata,
        sample=False,
    )

    assert out.shape == (5, n_genes)
    assert torch.all(out >= 0)
