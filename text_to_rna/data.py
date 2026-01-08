import numpy as np
import pyarrow as pa
import torch
import os
import json

from cytoolz import compose_left, curry, keyfilter, keymap, merge, pipe
from cytoolz.curried import concat
from functools import lru_cache

import math  # For math.isnan

CONTROLS = {
    "vehicle",
    "dmso",
    "untreated",
    "empty vector",
    "DMSO",
    "no treatment",
    "pbs",
    "mock",
    "ethanol",
    "vector",
    "scramble sirna",
    "scramble",
    "non-targeting sirna",
    "solvent",
    "rpmi",
    "media",
    "normoxia",
    "bsa",
    "water",
    "etoh",
    "air",
    "scrambled rna",
    "static",
    "dmem",
    "scrambled rna",
    "mock transfection",
    "mock-treated",
    "mock infected",
    "sham",
    "mock transfection",
    "mock transduction",
    "lacz",
    "luciferase",
}


GROUPED_COLUMNS = {
    "biological": [
        "sex",
        "cell_line_ontology_id",
        "cell_type_ontology_id",
        "disease_ontology_id",
        "tissue_ontology_id",
        "sample_type",
        "age_years",
        "ethnicity",
        "race",
        "genotype",
        "developmental_stage",
    ],
    "technical": [
        "library_selection",
        "library_layout",
        "platform",
        "study",
        "modality",
        "subject_identifier",
    ],
    "perturbation": [
        "perturbation_ontology_id",
        "perturbation_time",
        "perturbation_dose",
        "perturbation_type",
    ],
}

PASS_THROUGH_COLUMNS = [
    "tissue_name",
    "cell_type_name",
    "cell_line_name",
    "disease_name",
    "perturbation_name",
]

# Include pass-through columns in the table and metadata, but they are not part of latent groups
METADATA_COLUMNS = list(concat(GROUPED_COLUMNS.values()))
METADATA_COLUMNS.extend(PASS_THROUGH_COLUMNS)
METADATA_COLUMNS.sort()
TABLE_COLUMNS = (
    ["counts", "counts_ref", "experiment_accession"]
    + METADATA_COLUMNS
    + [col + "_ref" for col in METADATA_COLUMNS]
)
TABLE_COLUMNS.sort()


@lru_cache
def read_json(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, "r") as fp:
        out = json.load(fp)
    return out


def write_json(d: dict, path: str) -> None:
    with open(path, "w") as fp:
        json.dump(d, fp)
    return None


@lru_cache
def get_padding_dict(batch_size: int) -> dict:
    # used to fill in missing columns in metadata to ensure that all batches have the same columns regardless of modality
    return {k: [""] * batch_size for k in METADATA_COLUMNS}


def fix_age(age):
    age_fixed = []
    for string_val in age:
        if not string_val:  # Handles None or empty string
            age_fixed.append("")
        else:
            try:
                float_val = float(string_val)
                if math.isnan(float_val):
                    age_fixed.append(
                        ""
                    )  # Treat NaN as an empty string or other placeholder
                else:
                    age_fixed.append(str(int(round(float_val))))
            except (
                ValueError
            ):  # If string_val cannot be converted to float (e.g., "Unknown")
                age_fixed.append(
                    ""
                )  # Or handle as appropriate, e.g., pass string_val through
    return age_fixed


def process_expression(expression: np.ndarray) -> torch.Tensor:
    expression = np.vstack(expression)
    expression = torch.tensor(expression, dtype=torch.float32)
    return expression


def process_metadata(metadata, modality):
    batch_size = max(len(v) for v in metadata.values())
    padding_dict = get_padding_dict(batch_size)
    metadata = merge(padding_dict, metadata)
    metadata["modality"] = [modality] * batch_size

    if "age_years" in metadata.keys():
        metadata["age_years"] = fix_age(metadata["age_years"])
    return metadata


@curry
def collate(
    batch_: list[pa.RecordBatch], modality: str, dataset_name: str, source="pa"
) -> dict:
    if source == "pa":
        batch = pa.Table.from_batches(batch_)
    else:
        batch = pa.Table.from_pylist(batch_)

    if "counts" in batch.column_names:
        expression = process_expression(batch.column("counts").to_numpy())
        batch = batch.drop_columns(["counts"])
    else:
        expression = None

    if "counts_ref" in batch.column_names:
        reference = process_expression(batch.column("counts_ref").to_numpy())
        batch = batch.drop_columns(["counts_ref"])
    else:
        reference = None

    batch = batch.to_pydict()

    metadata = keyfilter(lambda k: "_ref" not in k, batch)
    metadata = process_metadata(metadata, modality)
    metadata_ref = keyfilter(lambda k: "_ref" in k, batch)

    if metadata_ref:
        metadata_ref = keymap(lambda k: k.replace("_ref", ""), metadata_ref)
        metadata_ref = process_metadata(metadata_ref, modality)

    collated_dict = {
        "expression": expression,
        "reference_expression": reference,
        "metadata": metadata,
        "reference_metadata": metadata_ref,
        "modality": modality,
        "dataset_name": dataset_name,
    }

    # validate_shapes(collated_dict)

    return collated_dict


def encode_multilabel(target_vals: dict, collated_dict: dict) -> dict:
    """
    Encodes labels to sparse COO tensors on CPU, suitable for MultilabelCCELoss

    - Assumes batch_size > 0 and is consistent across keys if used together later.
    - Creates tensors on CPU.
    - Metadata strings are split by '|'; each part is mapped to an integer class index.
    - Metadata parts not found in target_vals mappings are ignored.
    - Duplicate labels for the same sample are encoded once.
    - Determines num_classes based on the maximum index found in mappings + 1.

    Args:
        target_vals (dict): Defines label groups, metadata keys, and mappings
            from metadata string values to integer class indices.
        collated_dict (dict): Contains batch data, with
            collated_dict["metadata"][key] as a list of strings for the batch.

    Returns:
        dict: The collated_dict, updated with collated_dict["labels"],
              containing generated sparse target tensors (on CPU).
    """
    sparse_labels_output = {group: {} for group in target_vals.keys()}
    cpu_device = torch.device("cpu")

    # --- Outer loop over groups ---
    for group, keys_dict in target_vals.items():
        # --- Inner loop over keys within the group ---
        for key, vals_map in keys_dict.items():
            # Skip processing if the key is not found in the batch's metadata
            if key not in collated_dict.get("metadata", {}):
                continue

            key_num_classes = len(vals_map)
            metadata_batch_list = collated_dict["metadata"][key]
            current_key_batch_size = len(metadata_batch_list)  # Assumed > 0

            # Create an empty sparse tensor if no classes were defined
            if key_num_classes == 0:
                sparse_labels_output[group][key] = torch.sparse_coo_tensor(
                    torch.empty((2, 0), dtype=torch.long, device=cpu_device),
                    torch.empty(0, dtype=torch.float, device=cpu_device),
                    (current_key_batch_size, 0),
                )
                continue  # Skip to next key

            # --- Collect unique positive (sample_idx, class_idx) pairs ---
            unique_positive_labels = set()
            for i, metadata_item_str in enumerate(metadata_batch_list):
                # Split potential multi-labels by '|', ensure input is string
                parts = str(metadata_item_str).split("|")
                for part in parts:
                    cleaned_part = part.strip()
                    if not cleaned_part:
                        continue  # Skip empty parts

                    class_idx = vals_map.get(
                        cleaned_part
                    )  # Map string part to index (yields None if not found)

                    if class_idx is not None:  # Only process if mapping exists
                        # Validate type and range of mapped index
                        if not isinstance(class_idx, int):
                            raise TypeError(
                                f"[Group: {group}, Key: {key}] Mapped class index for '{cleaned_part}' must be an int, got {type(class_idx)}."
                            )
                        if not (0 <= class_idx < key_num_classes):
                            # This check ensures consistency with the calculated num_classes
                            raise ValueError(
                                f"[Group: {group}, Key: {key}] Class index {class_idx} for item '{cleaned_part}' is out of bounds (0 to {key_num_classes - 1})."
                            )

                        unique_positive_labels.add(
                            (i, class_idx)
                        )  # Add unique (sample, class) pair
            # --- End collecting pairs ---

            # --- Create sparse tensor ---
            if (
                not unique_positive_labels
            ):  # No positive labels found for this batch/key
                sparse_indices = torch.empty(
                    (2, 0), dtype=torch.long, device=cpu_device
                )
                sparse_values = torch.empty(0, dtype=torch.float, device=cpu_device)
            else:
                # Unzip the set into rows and columns, sorting for deterministic order
                sorted_labels = sorted(list(unique_positive_labels))
                indices_rows = [item[0] for item in sorted_labels]
                indices_cols = [item[1] for item in sorted_labels]
                sparse_indices = torch.tensor(
                    [indices_rows, indices_cols], dtype=torch.long, device=cpu_device
                )
                sparse_values = torch.ones(
                    len(indices_rows), dtype=torch.float, device=cpu_device
                )  # Use 1.0 for values

            sparse_target_tensor = torch.sparse_coo_tensor(
                sparse_indices,
                sparse_values,
                (current_key_batch_size, key_num_classes),
                device=cpu_device,
            ).coalesce()  # Ensure canonical sparse format
            # --- End sparse tensor creation ---

            sparse_labels_output[group][key] = sparse_target_tensor
        # --- End processing keys in group ---
    # --- End processing groups ---

    collated_dict["labels"] = sparse_labels_output
    return collated_dict
