from . import constants
import pandas as pd
from text_to_rna.data import read_json, write_json
import os
import torch
from functools import lru_cache

GENE_ORDER_JSON = "text_to_rna/gene_id_order.json"


@lru_cache
def get_gene_indices(modality):
    gene_order = read_json(GENE_ORDER_JSON)
    ref_genes = gene_order["union"]
    if modality in ["czi", "perturbseq"]:
        genes = gene_order["single_cell"]
    else:
        genes = gene_order[modality]

    # map genes to indices of genes in expanded matrix
    gene_to_idx = {gene: index for index, gene in enumerate(ref_genes)}

    # use mapping to find indices of genes in original gene order
    indices = [gene_to_idx[gene] for gene in genes]

    return indices


def align_gene_expression(expression: torch.Tensor, modality: str) -> torch.Tensor:
    assert isinstance(modality, str), "Modality must be a string"

    # Create the reindexed expression tensor
    reindexed_expression = torch.zeros(
        (expression.size(0), constants.TOTAL_GENES),
        device=expression.device,
        dtype=expression.dtype,
    )

    # Fill the reindexed tensor with existing gene data
    indices = get_gene_indices(modality)
    reindexed_expression[:, indices] = expression

    return reindexed_expression


def unalign_gene_expression(expression: torch.Tensor, modality: str) -> torch.Tensor:
    """
    Takes an expanded gene expression matrix and returns it to the original gene order containing only the observed genes
    """

    indices = get_gene_indices(modality)
    expression = expression[:, indices]

    return expression


def test_gene_alignment():
    for m in constants.MODALITIES.keys():
        indices = get_gene_indices(m)
        x = torch.arange(len(indices) * 6).reshape(6, len(indices)).float()
        y = unalign_gene_expression(align_gene_expression(x, m), m)
        assert torch.allclose(x, y), f"Failed for {m}"
    print("Gene alignment test passed!")


def build_gene_id_order():
    gene_id_order = {
        modality: (
            pd.read_csv(f"gene_info/{modality}.csv")
            .rename(
                columns={
                    "ensembl_gene_id": "gene_stable_id",
                }
            )
            .gene_stable_id.values.tolist()
        )
        for modality in constants.MODALITIES.keys()
    }

    all_genes = sorted(set().union(*gene_id_order.values()))
    gene_id_order["union"] = all_genes

    return gene_id_order


def build_gene_order(gene_id_order):
    ensembl_to_gene = (
        pd.read_csv("gene_info/ensembl_symbol.txt", sep="\t")
        .dropna()
        .drop_duplicates(subset="Ensembl ID(supplied by Ensembl)")
    )
    ensembl_to_gene = dict(
        zip(
            ensembl_to_gene["Ensembl ID(supplied by Ensembl)"],
            ensembl_to_gene["Approved symbol"],
        )
    )

    gene_order = {
        modality: [ensembl_to_gene.get(gene_id, gene_id) for gene_id in gene_ids]
        for modality, gene_ids in gene_id_order.items()
    }

    for k, v in gene_order.items():
        assert len(v) == len(set(v)), f"Duplicate gene names in {k} gene order"
        assert len(v) == len(gene_id_order[k]), f"Gene order length mismatch in {k}"
        print(f"{k}: {len(v)} genes")
    return gene_order


def build():
    gene_id_order = build_gene_id_order()
    gene_order = build_gene_order(gene_id_order)

    write_json(gene_order, "gene_order.json")
    write_json(gene_id_order, "gene_id_order.json")
    return None


if __name__ == "__main__":
    assert os.path.exists("gene_info"), "gene_info directory not found"
    assert not os.path.exists("gene_order.json"), "gene_order.json already exists"
    assert not os.path.exists("gene_id_order.json"), "gene_id_order.json already exists"

    build()
    test_gene_alignment()
