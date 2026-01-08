from . import constants
import math
import numpy as np
import os
import pandas as pd
import torch

from cytoolz import concat, thread_first, valmap
from cytoolz.curried import map, filter
import text_to_rna.data as data

NUMERIC_FIELDS = {"perturbation_time", "perturbation_dose"}
EPS = 1e-6

COUNTS_DIR = "text_to_rna/counts"
REPOSITORY_DIR = ""


def get_counts(field: str, modalities: list[str]) -> pd.DataFrame:
    counts = thread_first(
        modalities,
        map(lambda modality: f"{COUNTS_DIR}/{modality}_{field}.parquet"),
        filter(os.path.exists),
        list,
        pd.read_parquet,
    )

    if counts.empty:
        return pd.DataFrame({"count": []})

    if field == "age_years":
        counts["age_years"] = (
            counts.age_years.astype(float).round().astype(int).astype(str)
        )

    elif field in NUMERIC_FIELDS:
        counts[field] = counts[field].str.split(" ", expand=True)[1]

    counts = counts.groupby(field).sum()
    return counts


def encode_numeric(field: str, value: float, max_z: float = 6.0) -> float:
    value = np.log(value + EPS)
    scaling_factor = data.read_json(f"text_to_rna/{field}_znorm_factors.json")
    value = (value - scaling_factor["mean"]) / scaling_factor["std"]
    value = np.clip(value, -max_z, max_z)
    return value


def extract_numeric(string) -> tuple[float | None, str]:
    try:
        value, unit = string.split(" ")
        value = float(value)
        assert math.isfinite(value)
        assert value >= 0
    except:
        value = None
        unit = ""

    return value, unit


def weighted_std(values, weights):
    """
    Calculate the weighted standard deviation.

    Args:
        values: The values to calculate the standard deviation for.
        weights: The corresponding weights.

    Returns:
        The weighted standard deviation.
    """

    weighted_mean = np.average(values, weights=weights)
    weighted_variance = np.average((values - weighted_mean) ** 2, weights=weights)
    return weighted_mean, np.sqrt(weighted_variance)


def build_numeric_scale_factors(modalities: list[str], field: str) -> None:
    assert field in NUMERIC_FIELDS, f"{field} is not a numeric field"

    counts = thread_first(
        modalities,
        map(lambda modality: f"{COUNTS_DIR}/{modality}_{field}.parquet"),
        filter(os.path.exists),
        list,
        pd.read_parquet,
    )

    counts[field] = counts[field].str.split(" ", expand=True)[0].astype(float)
    counts = counts.groupby(field).sum().reset_index()
    values = np.log(counts[field].values + EPS)
    mean, std = weighted_std(values, counts["count"])

    data.write_json(
        {"mean": mean.tolist(), "std": std.tolist()},
        f"{REPOSITORY_DIR}{field}_znorm_factors.json",
    )


class EncodeMetadata(object):
    def __init__(self, modalities: list[str], min_count: int = 0, vectorize: dict = {}):
        # vectorize is a dict with {column_name : path_to_embedding_file}
        # the embedding file must be a pkl containing a pd.Series of key(column) : value(embedding)
        # all embeddings must be of the same length and all of the keys in the column must be unique

        self.columns = set(data.METADATA_COLUMNS)
        self.modalities = modalities
        self.min_count = min_count
        vocab = self.load_vocab(self.columns - set(vectorize.keys()) - {"modality"})

        if vectorize:
            missing_keys = self.load_vectors(vectorize)
            vocab.update(missing_keys)
        else:
            self.vectors = {}
            self.vector_dims = {}
            self.vector_len = 0

        vocab.update({"modality": set(self.modalities)})
        vocab.update({field: {field} | vocab[field] for field in NUMERIC_FIELDS})

        self.vocab = valmap(
            lambda terms: {k: v for v, k in enumerate(sorted(terms))}, vocab
        )

        # Precompute gene-to-index maps for each modality
        gene_order = data.read_json("text_to_rna/gene_order.json")
        self.gene_to_idx_maps = {}
        print(gene_order.keys(), "----------------------")
        for modality in self.modalities:
            if modality in ["czi", "perturbseq"]:
                genes = gene_order["single_cell"]
            else:
                genes = gene_order[modality]
            self.gene_to_idx_maps[modality] = {
                gene.upper(): i for i, gene in enumerate(genes)
            }

        # Other initializations...
        self.vocab = valmap(
            lambda terms: {k: v for v, k in enumerate(sorted(terms))}, vocab
        )

    def load_vocab(self, columns: set[str]) -> dict[str, set]:
        vocab = {
            column: set(
                get_counts(column, self.modalities)
                .query(f"count >= {self.min_count}")
                .index.values
            )
            for column in columns
            if column != "modality"
        }
        return vocab

    def load_vectors(self, vectorize: dict[str, str]) -> dict[str, set]:
        self.vectors = {}
        self.vector_dims = {}
        missing_keys = dict(zip(vectorize.keys(), [set()] * len(vectorize)))
        vocab = self.load_vocab(set(vectorize.keys()))

        ## Load the expected vocabulary for this column to compare with the embedding keys and identify any tokens that lack precomputed embeddings.
        for col, file_path in vectorize.items():
            print(file_path)
            print(f"Loading vectors for {col}, expecting {len(vocab[col])} keys")
            observed_keys = set()

            if isinstance(file_path, str):
                file_path = [file_path]

            self.vectors[col] = []
            self.vector_dims[col] = []

            for file in sorted(file_path):
                print(f"Loading {file} for {col}")
                df = pd.read_pickle(file)
                assert not df.empty, f"Empty embedding file: {file}"
                lens = df.str.len().unique()
                assert len(lens) == 1, (
                    f"Found embedding vecotrs of different lengths: {lens} in {file}"
                )

                df.index = df.index.astype(str)
                # assert not df.index.duplicated().any(), f"Found duplicate keys in {file}"

                self.vectors[col].append(df.to_dict())
                self.vector_dims[col].append(lens[0])
                observed_keys = set(df.index) | observed_keys

                print(f"Loaded {len(set(df.index))} keys from {file}")

            # identify missing vocab terms from embeddings to be encoded one-hot
            missing_keys[col] = vocab[col] - observed_keys
            print(f"Found {len(missing_keys[col])} missing keys in {col}")

        self.vector_len = sum(concat(self.vector_dims.values()))
        return missing_keys

    def __len__(self):
        return self.dimensions

    @property
    def dimensions(self) -> dict[str, int]:
        grouped_dims = dict()
        for group, cols in data.GROUPED_COLUMNS.items():
            grouped_dims[group] = sum(
                len(self.vocab[col]) for col in cols if col in self.vocab
            )

            vector_cols = set(cols) & set(self.vectors.keys())
            for col in vector_cols:
                grouped_dims[group] += sum(self.vector_dims[col])

        return grouped_dims

    def vectorize_col(self, values, col_name: str) -> torch.Tensor:
        missing = [np.zeros(dim) for dim in self.vector_dims[col_name]]

        vectors = []
        for value in values:
            current_value_str = value if value is not None else ""
            keys = current_value_str.split("|")
            sample_vector = []

            # iterate through vector dictionaries and lookup keys, filling in 0s when missing
            for vector_dict, miss in zip(self.vectors[col_name], missing):
                # sum the embeddings for multiple keys
                vector = np.sum([vector_dict.get(key, miss) for key in keys], axis=0)
                sample_vector.append(vector)

            # horizontally stack the embeddings for each sample
            vectors.append(np.hstack(sample_vector))

        vectors = np.array(vectors, dtype=np.float32)
        return torch.from_numpy(vectors)

    def vectorize_columns(self, batch: dict) -> dict:
        vectorized_keys = self.vectors.keys() & batch["metadata"].keys()

        if vectorized_keys:
            vectorized_cols = {
                col: self.vectorize_col(batch["metadata"][col], col)
                for col in vectorized_keys
            }
        else:
            vectorized_cols = {}

        return vectorized_cols

    def encode_column(self, batch: dict, column: str) -> torch.Tensor:
        x_s = []
        y_s = []
        vals = []

        for i, row in enumerate(batch["metadata"][column]):
            if not row:
                continue

            strings = row.split("|")

            if column in NUMERIC_FIELDS:
                numeric_value, numeric_unit = extract_numeric(strings[0])
                strings = [numeric_unit]
                if numeric_value is not None:
                    numeric_value = encode_numeric(column, numeric_value)
                    x_s.append(i)
                    y_s.append(self.vocab[column][column])
                    vals.append(numeric_value)

            idxes = [
                self.vocab[column][string]
                for string in strings
                if string in self.vocab[column]
            ]
            x_s.extend([i] * len(idxes))
            y_s.extend(idxes)
            vals.extend([1] * len(idxes))

        shape = (len(batch["metadata"][column]), len(self.vocab[column]))

        metadata_encoding = torch.sparse_coo_tensor(
            (x_s, y_s),
            vals,
            size=shape,
        )

        return metadata_encoding

    def encode_perturbation(self, batch: dict) -> torch.Tensor:
        indices = []

        if (
            "perturbation_ontology_id" not in batch["metadata"]
            or "modality" not in batch["metadata"]
        ):
            return torch.full(
                (len(batch["expression"]),), -1, dtype=torch.long
            )  # Or another suitable default value

        modality = batch["metadata"]["modality"][
            0
        ]  # Assuming all samples in a batch are from the same modality
        gene_to_idx_map = self.gene_to_idx_maps.get(modality)

        if gene_to_idx_map is None:
            return torch.full((len(batch["expression"]),), -1, dtype=torch.long)

        for gene_name, pert_type in zip(
            batch["metadata"]["perturbation_ontology_id"],
            batch["metadata"]["perturbation_type"],
        ):
            if isinstance(gene_name, str) and pert_type in {
                "crispr",
                "shrna",
                "overexpression",
                "sirna",
            }:
                idx = gene_to_idx_map.get(gene_name.upper(), -1)
            else:
                idx = -1
            indices.append(idx)

        return torch.tensor(indices, dtype=torch.long)

    def group_metadata(
        self, metadata_encoding: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        grouped_encoding = {}
        for group, cols in data.GROUPED_COLUMNS.items():
            tensors = [
                metadata_encoding[col] for col in cols if col in metadata_encoding
            ]

            if not tensors:
                continue

            if any(not getattr(t, "is_sparse", False) for t in tensors):
                tensors = [
                    t.to_dense() if getattr(t, "is_sparse", False) else t
                    for t in tensors
                ]
                grouped_encoding[group] = torch.cat(tensors, dim=1)
            else:
                grouped_encoding[group] = torch.cat(tensors, dim=1).coalesce()

        return grouped_encoding

    def __call__(self, batch: dict) -> dict:
        metadata_encoding = {
            column: self.encode_column(batch, column) for column in self.columns
        }

        if self.vectors.keys():
            vectorized_cols = self.vectorize_columns(batch)
            for col in vectorized_cols:
                metadata_encoding[col] = torch.cat(
                    (metadata_encoding[col].to_dense(), vectorized_cols[col]), dim=1
                )

        batch["modality_specific_perturbed_gene_idx"] = self.encode_perturbation(batch)
        batch["metadata_encoding"] = self.group_metadata(metadata_encoding)

        return batch


if __name__ == "__main__":
    for f in NUMERIC_FIELDS:
        print(f"Computing z-norm factors for {f}")
        build_numeric_scale_factors(list(constants.MODALITIES.keys()), f)
