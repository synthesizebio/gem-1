import os
import numpy as np
import argparse
import math
import torch
from cytoolz import concat, curry, merge, pipe, peek
from itertools import groupby
from cytoolz.curried import map, partition_all
from pytorch_lightning.utilities import move_data_to_device
from typing import Iterable, Optional, Iterator, Any
import text_to_rna.data as data
import pandas as pd
from text_to_rna.model.base_model import SynthesizeBioModel
from typing import Iterator, Tuple, Any
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as pa_ds
import warnings

torch.no_grad()
BATCH_SIZE = 128
OUTPUT_BATCH_SIZE = 1024
import os
import ast
from typing import Optional
from omegaconf import OmegaConf
import numpy as np


def load_model(
    modality: str,
    checkpoint_file: str = "last.ckpt",
    base_dir: str = "data/model_files",
    strict: bool = True,
    map_location: Optional[str] = None,
):
    """
    Load a local model checkpoint and its saved YAML config (no W&B dependency).

    Args:
        modality (str): One of {"bulk", "single_cell"}.
        checkpoint_file (str): Checkpoint file name (default: "last.ckpt").
        base_dir (str): Directory containing saved model subfolders.
        strict (bool): Whether to enforce exact state_dict matching.
        map_location (Optional[str]): torch map_location for device remapping.
    """
    modality = str(modality).lower().strip()
    if modality not in {"bulk", "single_cell"}:
        raise ValueError(
            f"Unsupported modality '{modality}'. Use 'bulk' or 'single_cell'."
        )

    model_dir = os.path.join(base_dir, modality)
    ckpt_path = os.path.join(model_dir, checkpoint_file)
    cfg_path = os.path.join(model_dir, "config.yaml")

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"❌ Checkpoint not found: {ckpt_path}")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"❌ Config not found: {cfg_path}")

    if modality == "bulk":
        from text_to_rna.model.autoencoder.unified_vae import UnifiedVAE as ModelClass
    else:
        from text_to_rna.model.autoencoder.unified_vae_cce import (
            UnifiedVAECCE as ModelClass,
        )

    model = ModelClass.load_from_checkpoint(
        checkpoint_path=ckpt_path, strict=strict, map_location=map_location
    )

    print(f"✅ Successfully loaded {modality} model.")
    return model


def iterate_study_predictions(
    source_parquet_path: str, prediction_parquet_path: str
) -> Iterator[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Iterates through studies from a source Parquet file, retrieves corresponding
    predictions from a partitioned Parquet dataset, and yields data for each study.

    Args:
        source_parquet_path: Path to the source Parquet file containing original data.
        prediction_parquet_path: Path to the root of the Parquet dataset containing
                                 predictions, partitioned by 'study'.

    Yields:
        A tuple for each study:
        (
            source_df: Dataframe of source metadata and 'counts' for observed counts
            predicted_df: DataFrame of metadata and 'counts_pred' for predicted counts.
        )
    """

    print(f"Reading source Parquet: {source_parquet_path}")
    try:
        source_df = pd.read_parquet(source_parquet_path)
    except Exception as e:
        print(f"Error reading source Parquet file {source_parquet_path}: {e}")
        return

    if "study" not in source_df.columns:
        print("Error: 'study' column not found in source Parquet file.")
        return

    drop_col = [col for col in source_df.columns if col not in data.TABLE_COLUMNS]
    source_df.drop(columns=drop_col, inplace=True)

    # Sort by study to ensure groupby works correctly
    source_df.sort_values(by="study", inplace=True)

    for study_id, group_df in source_df.groupby("study"):
        print(f"\nProcessing study: {study_id}")

        # Construct path to the prediction data for the current study
        # Assumes partitioning like /prediction_parquet_path/study=STUDY_ID/
        study_prediction_path = os.path.join(
            prediction_parquet_path, f"study={study_id}"
        )

        if os.path.exists(study_prediction_path) and os.path.isdir(
            study_prediction_path
        ):
            print(
                f"Reading predictions for study {study_id} from {study_prediction_path}"
            )
            # Read all Parquet files within the study's partition directory
            pred_df = pd.read_parquet(study_prediction_path)
            pred_df["study"] = (
                study_id  # Ensure 'study' column is present in predictions
            )
        else:
            print(
                f"Warning: Prediction path not found for study {study_id}: {study_prediction_path}"
            )
            continue

        yield (
            group_df[group_df.experiment_accession.isin(pred_df.experiment_accession)],
            pred_df,
        )


def expand_dict(row: dict[str, Any]) -> dict[str, Any]:
    """
    Expands an input dictionary representing a single sample/row.

    Ensures that all common metadata fields (defined in `data.METADATA_COLUMNS`)
    are present in the output dictionary, defaulting to an empty string if a field
    is missing from the input row.

    Additionally, it standardizes the 'age_years' field:
    - Converts NaN, None, or unparseable 'age_years' values to an empty string.
    - Converts valid numeric 'age_years' values to a string representation of the rounded integer.

    Args:
        row: A dictionary representing a single row of data.

    Returns:
        A new dictionary with all common metadata fields and standardized 'age_years'."""
    # Start with all keys from the original row
    expanded_row = row.copy()

    # Ensure all METADATA_COLUMNS are present, defaulting to "" if not in original row
    for k in data.METADATA_COLUMNS:
        if k not in expanded_row:
            expanded_row[k] = ""

    # Now, specifically process 'age_years' in the expanded_row
    if "age_years" in expanded_row:
        val = expanded_row["age_years"]

        if val is None:
            expanded_row["age_years"] = ""
        elif isinstance(val, (float, np.floating)) and math.isnan(
            val
        ):  # Check for float NaN
            expanded_row["age_years"] = ""
        else:
            # For all other types (including strings, ints, etc.)
            try:
                s_val = str(val).strip()
                if not s_val or s_val.lower() in [
                    "nan",
                    "unknown",
                    "none",
                    "null",
                    "<na>",
                ]:
                    expanded_row["age_years"] = ""
                else:
                    float_val = float(s_val)
                    if math.isnan(float_val):
                        expanded_row["age_years"] = ""
                    else:
                        expanded_row["age_years"] = str(int(round(float_val)))
            except (ValueError, TypeError):
                expanded_row["age_years"] = ""

    return expanded_row


@torch.no_grad()
def write_parquet(
    output_path: str,
    data_iterable: Iterable[dict[str, Any]],
    batch_size: int = OUTPUT_BATCH_SIZE,
):
    """
    Writes an iterable of dictionaries to a multi-file Parquet dataset,
    partitioned by the 'study' column.

    The function groups the input data by the 'study' field. For each study,
    it writes the data in chunks specified by `batch_size` to avoid creating
    excessively large individual Parquet files or holding too much data in memory.
    The schema for the Parquet dataset is inferred from the first processed batch
    of the first study and is then used for all subsequent writes.

    Args:
        output_path: The root directory path where the partitioned Parquet dataset will be written.
                     A subdirectory will be created for each unique study ID (e.g., output_path/study=A/).
        data_iterable: An iterable of dictionaries, where each dictionary represents a row of data.
                       It's crucial that this iterable yields data effectively sorted by 'study'
                       for `itertools.groupby` to work as intended for partitioning.
        batch_size: The number of rows to include in each Parquet file chunk written for a study.

    Returns:
        None. Prints status messages to the console."""
    if os.path.isdir(output_path):
        warnings.warn(
            f"Output path '{output_path}' already exists; "
            "any new predictions will be concatenated into it.",
            UserWarning,
        )

    os.makedirs(output_path, exist_ok=True)

    first_table_processed = False
    dataset_schema = None

    # Group data by study. This assumes data_iterable is sorted or effectively sorted by 'study'.
    # If not, groupby will create many small groups for interleaved studies.
    for study_id, study_rows_iterable in groupby(
        data_iterable, key=lambda row: row.get("study")
    ):
        if study_id is None:
            print(
                "Warning: Found rows with no 'study' ID. These rows will be skipped for partitioning."
            )
            # Optionally, collect these rows and write them to a separate unpartitioned file or log them.
            for _ in study_rows_iterable:
                pass  # Consume the iterator
            continue

        # Process each study in chunks defined by batch_size
        for batch_of_dicts in pipe(study_rows_iterable, partition_all(batch_size)):
            if not batch_of_dicts:
                continue

            # Convert list of dicts in the current batch to a pandas DataFrame
            # Ensure batch_of_dicts is materialized if it's an iterator from partition_all
            current_batch_list = list(batch_of_dicts)
            if not current_batch_list:
                continue

            df_batch = pd.DataFrame(current_batch_list)

            if df_batch.empty:
                continue

            current_arrow_table = None
            if not first_table_processed:
                if "study" not in df_batch.columns:
                    # This should ideally not happen if we are grouping by study_id from get("study")
                    raise ValueError(
                        "The 'study' column is required for partitioning but was not found in the data."
                    )
                current_arrow_table = pa.Table.from_pandas(
                    df_batch, preserve_index=False
                )
                if current_arrow_table.num_rows > 0:
                    dataset_schema = current_arrow_table.schema
                    first_table_processed = True
            else:
                # For subsequent batches, create pyarrow.Table using the established schema
                current_arrow_table = pa.Table.from_pandas(
                    df_batch, schema=dataset_schema, preserve_index=False
                )

            if current_arrow_table is not None and current_arrow_table.num_rows > 0:
                try:
                    pq.write_to_dataset(
                        current_arrow_table,
                        root_path=output_path,
                        schema=dataset_schema,
                        partition_cols=["study"],
                        compression="snappy",
                        use_threads=True,
                        existing_data_behavior="overwrite_or_ignore",
                    )
                except Exception as e:
                    print(
                        f"Error writing a batch for study '{study_id}' to dataset {output_path}: {e}"
                    )
                    # Decide on error handling: return None, re-raise, or continue
                    return None

    if not first_table_processed:
        print(f"No data was generated to write to {output_path}.")
        return None

    print(f"Data successfully written to {output_path}, partitioned by 'study'.")
    return None


def stream_parquet_study(input_root: str, study_id: str) -> Iterator[dict[str, Any]]:
    """Stream rows for a specific study with predicate pushdown and column trimming.

    - Ensures required metadata defaults and modality
    - Trims to data.TABLE_COLUMNS
    - Sorts by technical columns for contiguous grouping
    """
    try:
        dset = pa_ds.dataset(input_root, format="parquet")
        filt = pa_ds.field("study") == study_id
        cols = list(data.TABLE_COLUMNS)
        tbl = dset.to_table(
            filter=filt, columns=[c for c in cols if c in dset.schema.names]
        )
        df = tbl.to_pandas()
    except Exception:
        df = pd.read_parquet(input_root)
        if "study" in df.columns:
            df = df[df["study"].astype(str) == str(study_id)]

    for missing in data.METADATA_COLUMNS:
        if missing not in df.columns:
            df[missing] = ""
    if "modality" not in df.columns:
        df["modality"] = "bulk"

    drop = [c for c in df.columns if c not in data.TABLE_COLUMNS]
    if drop:
        df = df.drop(columns=drop)

    technical_cols = [c for c in data.GROUPED_COLUMNS["technical"] if c in df.columns]
    if technical_cols:
        df = df.sort_values(by=technical_cols, ignore_index=True)

    for row in df.to_dict(orient="records"):
        yield row


def stream_parquets(input_path: str, replicates: int = 1) -> Iterator[dict[str, Any]]:
    """
    Reads Parquet data from a path. If `input_path` is a dataset root partitioned by
    `study=...`, iterates each study partition sequentially and yields rows from each
    partition to keep memory bounded. Otherwise, reads the single path.

    For each loaded table, injects a 'study' column if missing by parsing hive-style
    partition (study=<ID>) from the path, drops non-table columns, then returns an
    iterator over rows. If a 'study' column is present, rows are sorted by study for
    contiguous grouping downstream.

    Each row is processed by `expand_dict` to ensure consistent metadata fields.

    Args:
        input_path: Path to a directory containing Parquet files or a single Parquet file.
        replicates: The number of times to replicate each row.

    Returns:
        An iterator yielding dictionaries, where each dictionary represents a processed row."""
    # Memory-safe: if input_path is a root with study= partitions, iterate them
    if os.path.isdir(input_path):
        study_dirs = [
            os.path.join(input_path, d)
            for d in sorted(os.listdir(input_path))
            if d.startswith("study=") and os.path.isdir(os.path.join(input_path, d))
        ]
        if study_dirs:
            for sp in study_dirs:
                yield from stream_parquets(sp, replicates=replicates)
            return
        # If this is a leaf study directory containing Parquet files, stream one file at a time
        parquet_files = [
            os.path.join(input_path, f)
            for f in sorted(os.listdir(input_path))
            if f.endswith(".parquet") and os.path.isfile(os.path.join(input_path, f))
        ]
        if parquet_files:
            for pf in parquet_files:
                yield from stream_parquets(pf, replicates=replicates)
            return

    print(f"Reading Parquet files from {input_path}...")
    # Project only needed columns to reduce memory pressure
    try:
        # Prefer reading with column projection when possible
        df = pd.read_parquet(
            input_path,
            engine="pyarrow",
            columns=[
                col
                for col in data.TABLE_COLUMNS
                # Let the engine ignore missing columns; pandas will validate
            ],
        )
    except Exception:
        # Fallback to reading all columns if projection fails for any reason
        df = pd.read_parquet(input_path, engine="pyarrow")
    # Inject study from hive-partitioned path if missing
    if "study" not in df.columns:
        study_id = None
        for part in str(input_path).split("/"):
            if part.startswith("study="):
                study_id = part.split("=", 1)[1]
                break
        if study_id is not None:
            df["study"] = study_id
    drop_col = [col for col in df.columns if col not in data.TABLE_COLUMNS]
    if drop_col:
        df.drop(columns=drop_col, inplace=True)
    # Ensure rows are contiguous by technical grouping so itertools.groupby downstream
    # yields groups that contain both controls and perturbed rows for the same context
    technical_cols = [c for c in data.GROUPED_COLUMNS["technical"] if c in df.columns]
    if technical_cols:
        df.sort_values(by=technical_cols, inplace=True, ignore_index=True)
    print(f"Loaded {len(df)} rows from Parquet files in {input_path}.")

    all_rows_list = df.to_dict(orient="records")
    if replicates > 1:
        all_rows_list = [row for row in all_rows_list for _ in range(replicates)]
    # Yield rows for downstream processing
    for row in (expand_dict(row) for row in all_rows_list):
        yield row


def ref_parquet_to_parquet(
    modality: str,
    input_path: str,
    output_path: str,
    mode: str,
    batch_size: int = BATCH_SIZE,
    output_batch_size: int = OUTPUT_BATCH_SIZE,
    strict: bool = True,
    total_count: int = 10_000_000,
    fixed_total_count: bool = False,
    replicates: int = 1,
    model: Optional[SynthesizeBioModel] = None,
    classification_ready_output: bool = False,
):
    """
    Predicts perturbation effects using observed reference data.

    This function loads a specified model, reads input Parquet files containing
    experimental data (including control and perturbed samples), processes this data
    to predict perturbation outcomes, and writes the results to output Parquet files.
    The input data is grouped by technical factors (including 'study'), and
    perturbation predictions are made within these groups using control samples
    from the same group as references.

    Args:
        modality: The data modality (e.g., "bulk", "single_cell").
        input_path: Path to the input Parquet files or directory.
        output_path: Path to write the output Parquet dataset.
        batch_size: Batch size for model prediction.
        output_batch_size: Batch size for writing output Parquet files."""
    loaded_model = load_model(modality=modality, strict=strict).eval()
    raw_data_stream = stream_parquets(input_path, replicates=replicates)

    # Define a key function for grouping by technical metadata columns
    technical_group_key = lambda row: tuple(
        row.get(k, "N/A_TECH_KEY") for k in data.GROUPED_COLUMNS["technical"]
    )
    it = pipe(
        raw_data_stream,
        curry(groupby, key=technical_group_key),
        map(
            lambda group_tuple: predict_perturbation_ref_study(
                rows=list(group_tuple[1]),  # Materialize the group for this call
                model=loaded_model,
                modality=modality,
                mode=mode,
                batch_size=batch_size,
                total_count=total_count,
                fixed_total_count=fixed_total_count,
                classification_ready_output=classification_ready_output,
            )
        ),
        concat,
    )

    write_parquet(output_path, it, output_batch_size)


def parquet_to_parquet(
    modality: str,
    input_path: str,
    output_path: str,
    mode: str,
    use_common_technical: bool = True,
    batch_size: int = BATCH_SIZE,
    output_batch_size: int = OUTPUT_BATCH_SIZE,
    strict: bool = True,
    total_count: int = 10_000_000,
    fixed_total_count: bool = False,
    replicates: int = 1,
    model: Optional[SynthesizeBioModel] = None,
):
    """
    Predicts gene expression from metadata.

    Loads a model, reads input Parquet files containing metadata,
    predicts expression for each sample, and writes the predictions
    (including original metadata and predicted counts) to output Parquet files.

    Args:
        modality: Data modality.
        input_path: Path to input Parquet files.
        output_path: Path for output Parquet dataset.
        use_common_technical: If True, uses `predict_studies` for common technical factor modeling.
                              Otherwise, uses `predict_stream` for independent sample prediction.
        batch_size: Batch size for model prediction.
        output_batch_size: Batch size for writing output Parquet files."""
    loaded_model = load_model(modality=modality, strict=strict).eval()
    data_stream = stream_parquets(input_path, replicates=replicates)

    if use_common_technical:
        predictions = predict_studies(
            rows=data_stream,
            model=loaded_model,
            modality=modality,
            batch_size=batch_size,
            mode=mode,
            total_count=total_count,
            fixed_total_count=fixed_total_count,
        )
    else:
        predictions = predict_stream(
            rows=data_stream,
            model=loaded_model,
            modality=modality,
            batch_size=batch_size,
            mode=mode,
            total_count=total_count,
            fixed_total_count=fixed_total_count,
        )

    write_parquet(output_path, predictions, output_batch_size)


def rowify_batch(batch_data: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Converts a batch of data from a columnar format (tensors, lists of metadata)
    into a list of row-oriented dictionaries.

    This function is essential for transforming the output of model prediction steps,
    which are typically batched tensors and associated metadata lists, into a
    format suitable for writing to tabular storage like Parquet.

    It infers the batch size from the first tensor encountered or from consistent
    list lengths in the metadata. It then iterates `batch_size` times, constructing
    a dictionary for each row by picking the i-th element from each column/list.

    Args:
        batch_data: A dictionary where keys are column names and values are
                    torch.Tensors or lists representing a batch of data for that column.
                    Nested dictionaries (e.g., for latents) are also handled."""
    processed_data = {}
    batch_size = 0

    for key, value in batch_data.items():
        if isinstance(value, torch.Tensor):
            if batch_size == 0:
                batch_size = value.shape[0]
            elif value.shape[0] != batch_size:
                raise ValueError(
                    f"Inconsistent batch sizes found for key {key}: expected {batch_size}, got {value.shape[0]}"
                )
            processed_data[key] = value.detach().cpu().numpy().tolist()
        elif isinstance(
            value, dict
        ):  # e.g. latents {group: tensor} or classifier_probs
            inner_dict = {}
            for inner_key, inner_value in value.items():
                if isinstance(inner_value, torch.Tensor):
                    if batch_size == 0:
                        batch_size = inner_value.shape[0]
                    elif inner_value.shape[0] != batch_size:
                        raise ValueError(
                            f"Inconsistent batch sizes for {key}.{inner_key}: expected {batch_size}, got {inner_value.shape[0]}"
                        )
                    inner_dict[inner_key] = inner_value.detach().cpu().numpy().tolist()
                elif isinstance(
                    inner_value, list
                ):  # For classifier_probs_to_dicts output
                    if batch_size == 0:
                        batch_size = len(inner_value)
                    elif len(inner_value) != batch_size:
                        raise ValueError(
                            f"Inconsistent batch sizes for {key}.{inner_key}: expected {batch_size}, got {len(inner_value)}"
                        )
                    inner_dict[inner_key] = inner_value
                else:
                    inner_dict[inner_key] = inner_value
            processed_data[key] = inner_dict
        else:  # Already list (e.g. metadata from collate) or scalar
            processed_data[key] = value
            if (
                isinstance(value, list)
                and batch_size == 0
                and value
                and isinstance(value[0], (str, int, float, dict))
            ):  # Infer batch_size from metadata lists
                current_list_len = len(value)
                is_consistent_list_batch = all(
                    isinstance(v_list, list) and len(v_list) == current_list_len
                    for v_list in batch_data.values()
                    if isinstance(v_list, list)
                )
                if is_consistent_list_batch:
                    batch_size = current_list_len

    if batch_size == 0 and processed_data:
        return [processed_data]

    output_rows = []
    for i in range(batch_size):
        row = {}
        for key, value_col in processed_data.items():
            if isinstance(value_col, list) and len(value_col) == batch_size:
                row[key] = value_col[i]
            elif isinstance(
                value_col, dict
            ):  # e.g. latents or classifier_probabilities
                row[key] = {
                    k_inner: v_inner_list[i]
                    for k_inner, v_inner_list in value_col.items()
                    if isinstance(v_inner_list, list)
                    and len(v_inner_list) == batch_size
                }
            else:  # Scalar metadata shared across batch or other non-batched info
                row[key] = value_col
        output_rows.append(row)
    return output_rows


@torch.no_grad()
def predict_stream(
    rows: Iterable[dict[str, Any]],
    model: "SynthesizeBioModel",  # Forward declaration for type hint
    modality: str,
    batch_size: int = BATCH_SIZE,
    mode: str = "sample generation",
    conditioning: Optional[tuple[str, ...]] = (),
    total_count: int = 10_000_000,
    fixed_total_count: bool = False,
    deterministic_latents: bool = False,
) -> Iterable[dict[str, Any]]:
    """
    Generic stream processing function for making predictions with the model.

    Takes an iterable of input rows, processes them in batches, and yields
    an iterable of output rows containing predictions.

    Args:
        rows: An iterable of input data dictionaries.
        model: The pre-loaded, evaluated SynthesizeBioModel instance.
        modality: The data modality (e.g., "bulk").
        batch_size: Number of samples to process in each model batch.
        mode: Prediction mode, typically "sample generation" (for sampling from the
              decoder) or "mean estimation" (for using the mean of the decoder output).
        conditioning: A tuple of metadata groups (e.g., ("technical", "biological"))
                      to condition on during prediction. If empty, no explicit conditioning
                      is applied beyond what's inherent in the input metadata.

    Returns:
        An iterable of dictionaries, where each dictionary is an output row
        containing original metadata, predicted counts ('counts_pred'), and latent variables."""
    sample_decoder = mode == "sample generation"
    it = pipe(
        rows,
        partition_all(batch_size),
        map(
            curry(
                data.collate,
                modality=modality,
                dataset_name="prediction_dataset",
                source="pylist",
            )
        ),
        map(model.collate("val")),
        map(curry(move_data_to_device, device=model.device)),
        map(
            lambda batch: model.predict_step(
                batch=batch,
                batch_idx=0,
                conditioning=conditioning,
                sample_decoder=sample_decoder,
                total_count=total_count,
                fixed_total_count=fixed_total_count,
                deterministic_latents=deterministic_latents,
            )
        ),
        map(rowify_batch),  # Convert batched output to list of row dicts
        concat,  # Flatten list of lists into a single iterable of rows
    )
    return it


def predict_studies(
    rows: Iterable[dict[str, Any]],
    modality: str,
    model: SynthesizeBioModel,
    batch_size: int,
    mode: str,
    total_count: int = 10_000_000,
    fixed_total_count: bool = False,
):
    """
    Predicts expression for studies, applying a common technical factor model.

    Groups input rows by technical metadata factors. For each group (study),
    it uses `predict_common_technical` to generate predictions, assuming
    samples within the same technical group share common technical variations.

    Args:
        rows: An iterable of input data dictionaries.
        modality: The data modality.
        model: The pre-loaded, evaluated SynthesizeBioModel instance.
        batch_size: Batch size for model prediction.
        mode: Prediction mode ("sample generation" or "mean estimation").

    Returns:
        An iterable of dictionaries, each representing a predicted sample."""

    predict = curry(
        predict_common_technical,
        modality=modality,
        model=model,
        batch_size=batch_size,
        mode=mode,
        total_count=total_count,
        fixed_total_count=fixed_total_count,
    )

    technical_group_key = lambda row: tuple(
        row.get(k) for k in data.GROUPED_COLUMNS["technical"]
    )
    grouper = curry(groupby, key=technical_group_key)
    it = pipe(
        rows,
        grouper,
        map(lambda x: predict(x[1])),
        concat,  # Flatten the iterables from each group
    )
    return it


def predict_common_technical(
    study_data: Iterable[dict[str, Any]],
    model: SynthesizeBioModel,
    modality: str,
    batch_size: int = BATCH_SIZE,
    mode: str = "sample generation",
    total_count: int = 10_000_000,
    fixed_total_count: bool = False,
    deterministic_latents: bool = False,
) -> Iterable[dict[str, Any]]:
    """
    Predicts expression for a single study (or technical group), modeling a
    common technical effect across its samples.

    It first makes a prediction for an example sample from the study without
    technical conditioning. This prediction is then used as a 'reference'
    (specifically, its `counts_pred` becomes `counts_ref`) for all other
    samples in the study. Subsequent predictions for the study's samples
    are then conditioned on "technical" factors, using this common reference.

    Args:
        study_data: An iterable of dictionaries, all belonging to the same study/technical group.
        model: The SynthesizeBioModel instance.
        modality: Data modality.
        batch_size: Batch size for prediction.
        mode: Prediction mode ("sample generation" or "mean estimation").

    Returns:
        An iterable of dictionaries, each representing a predicted sample for the study."""
    example, study_data = peek(study_data)

    example_pred = predict_stream(
        [example],
        model=model,
        modality=modality,
        batch_size=batch_size,
        mode=mode,
        conditioning=tuple(),
        total_count=total_count,
        fixed_total_count=fixed_total_count,
        deterministic_latents=deterministic_latents,
    )
    example_pred = next(example_pred)

    # Use the prediction from the example sample as the reference for all samples in this study
    study_data = map(
        lambda x: merge(x, {"counts_ref": example_pred["counts_pred"]}), study_data
    )

    # Predict for all samples in the study, now with technical conditioning using the common reference
    preds = predict_stream(
        study_data,
        model=model,
        modality=modality,
        batch_size=batch_size,
        mode=mode,
        conditioning=("technical",),
        total_count=total_count,
        fixed_total_count=fixed_total_count,
        deterministic_latents=deterministic_latents,
    )

    return preds


@torch.no_grad()
def classify_samples(
    model: "SynthesizeBioModel",  # Forward declaration
    modality: str,
    data_iterator: Iterable[dict[str, Any]],
    batch_size: int = BATCH_SIZE,
    mode: str = "sample generation",
    deterministic_latents: bool = False,
) -> Iterable[dict[str, Any]]:
    """
    Classifies samples based on their expression data using the model's learned classifiers.

    Takes an iterator of data dictionaries, processes them in batches,
    encodes expression to latents, passes latents through classifiers,
    and yields rows containing original metadata, classifier predictions/probabilities,
    and sampled latent variables.

    Args:
        model: The pre-loaded, evaluated SynthesizeBioModel instance.
        modality: The data modality.
        data_iterator: An iterable of input data dictionaries.
        batch_size: Number of samples to process in each model batch.

    Returns:
        An iterable of dictionaries, each representing a classified sample."""

    sample_decoder = mode == "sample generation"
    it = pipe(
        data_iterator,
        partition_all(batch_size),
        map(
            curry(
                data.collate,
                modality=modality,
                dataset_name="classification_dataset",
                source="pylist",
            )
        ),
        map(model.collate("val")),  # model.collate typically handles metadata encoding
        map(
            curry(move_data_to_device, device=model.device)
        ),  # Move data to model's device
        map(
            lambda batch: model.predict_vae(
                batch,
                sample_decoder=sample_decoder,
                deterministic_latents=deterministic_latents,
            )
        ),
        map(rowify_batch),  # Convert batch to list of row dicts
        concat,  # Flatten list of lists
    )
    return it


def classify_parquet(
    modality: str,
    input_path: str,
    output_path: str,
    mode: str = "sample generation",
    batch_size: int = BATCH_SIZE,
    output_batch_size: int = OUTPUT_BATCH_SIZE,
    strict: bool = True,
    replicates: int = 1,
    model: Optional[SynthesizeBioModel] = None,
):
    """
    Reads data from input Parquet files, classifies each sample using the specified model,
    and writes the classification results (including metadata, predictions, probabilities,
    and latents) to an output Parquet dataset.

    Args:
        modality: The data modality.
        input_path: Path to the input Parquet files or directory.
        output_path: Path to write the output Parquet dataset.
        batch_size: Batch size for model classification.
        output_batch_size: Batch size for writing output Parquet files.
    """
    loaded_model = load_model(modality=modality, strict=strict).eval()
    data_stream = stream_parquets(input_path, replicates=replicates)
    classified_stream = classify_samples(
        model=loaded_model,
        data_iterator=data_stream,
        modality=modality,
        batch_size=batch_size,
        mode=mode,
    )
    write_parquet(output_path, classified_stream, output_batch_size)
    return None


def to_ref_conditioned_pert_df(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Conditions perturbed samples with reference (control) samples from a DataFrame.

    This function takes a DataFrame containing both control and perturbed samples
    (assumed to be from a single study or technical group). It identifies control
    samples and merges their information onto perturbed samples based on matching
    technical and biological metadata columns. The merged DataFrame contains
    original perturbed sample data alongside corresponding reference data (suffixed with '_ref').

    Args:
        df: A pandas DataFrame with samples from a single technical/study group.
            Must contain 'perturbation_type' and relevant metadata columns for joining.
    Returns:
        A pandas DataFrame containing perturbed samples joined with their matched controls. Returns empty if no valid conditioning possible."""
    controls = df.query("perturbation_type == 'control'")
    perturbed = df.query("perturbation_type != 'control'")
    grouped = perturbed.groupby(data.GROUPED_COLUMNS["perturbation"])
    join_cols = data.GROUPED_COLUMNS["technical"] + data.GROUPED_COLUMNS["biological"]

    # Add perturbation_time to join keys if there's more than one common time point between controls and perturbations.
    if (
        len(
            set(controls.perturbation_time.unique())
            & set(perturbed.perturbation_time.unique())
        )
        > 1
    ):
        join_cols = join_cols + ["perturbation_time"]

    if controls.empty or perturbed.empty:
        return pd.DataFrame()

    out = []

    for group_cols, group in grouped:
        ref_conditioned = pd.merge(
            group,
            controls,
            on=join_cols,
            suffixes=("", "_ref"),
        )
        # If an explicit reference experiment hint is provided, restrict to 1:1 pairing
        """
        Within each technical group, this filter keeps:
        - Controls: they have no hint, so they pass.
        - Synthetics: only those whose ref_experiment_accession_hint equals the chosen experiment_accession_ref; mismatches are dropped.
        """
        if (
            "ref_experiment_accession_hint" in ref_conditioned.columns
            and "experiment_accession_ref" in ref_conditioned.columns
        ):
            ref_conditioned = ref_conditioned[
                ref_conditioned["ref_experiment_accession_hint"].isna()
                | (
                    ref_conditioned["ref_experiment_accession_hint"]
                    == ref_conditioned["experiment_accession_ref"]
                )
            ]
        # Drop the hint column from output rows
        if "ref_experiment_accession_hint" in ref_conditioned.columns:
            ref_conditioned = ref_conditioned.drop(
                columns=["ref_experiment_accession_hint"]
            )
        for col in join_cols:
            ref_conditioned[f"{col}_ref"] = ref_conditioned[col]

        out.append(ref_conditioned)

    return pd.concat(out, ignore_index=True)


def to_ref_conditioned_pert(
    rows_iterable: Iterable[dict[str, Any]],
):
    """
    Prepares perturbed samples for reference-conditioned prediction.

    If input rows already contain 'counts_ref' (indicating pre-joined reference data),
    they are passed through. Otherwise, it converts the iterable of rows to a DataFrame,
    uses `to_ref_conditioned_pert_df` to join perturbed samples with controls,
    and then converts the resulting DataFrame back to an iterable of dictionaries.

    Args:
        rows_iterable: An iterable of dictionaries, each representing a sample.
                       Assumed to be from a single study/technical group.
    Returns:
        An iterable of dictionaries, where perturbed samples are conditioned with
        reference data."""
    # Materialize the iterable to a list to allow inspection and DataFrame conversion
    rows_list = list(rows_iterable)

    if not rows_list:
        return iter([])  # Return an empty iterator if input is empty

    # Check the first row for 'counts_ref'
    if "counts_ref" in rows_list[0]:
        # If the first row has 'counts_ref', we assume it is already prejoined
        return iter(rows_list)  # Return an iterator over the original list

    df = pd.DataFrame(rows_list)
    df = to_ref_conditioned_pert_df(df)
    if df.empty:
        return iter([])

    return iter(df.to_dict(orient="records"))


def predict_perturbation_ref_study(
    rows: Iterable[dict[str, Any]],
    model: SynthesizeBioModel,
    modality: str,
    mode: str,
    batch_size: int = BATCH_SIZE,
    total_count: int = 10_000_000,
    fixed_total_count: bool = False,
    classification_ready_output: bool = False,
    deterministic_latents: bool = False,
) -> Iterable[dict[str, Any]]:
    """
    Predicts perturbation effects for a single study/technical group using observed reference data.

    This function processes an iterable of rows assumed to belong to the same
    technical group. It separates control and perturbed samples.
    - Control samples are passed through (with 'counts' renamed to 'counts_pred').
    - Perturbed samples are conditioned using `to_ref_conditioned_pert` (which
      joins them with appropriate controls from the group).
    - Predictions are then made for these conditioned perturbed samples.
    The output stream yields the original control samples first, followed by the
    predicted outcomes for the perturbed samples. Latent variables are removed from predictions.

    Args:
        rows: An iterable of dictionaries, all samples from a single study/technical group.
        model: The SynthesizeBioModel instance.
        modality: Data modality.
        mode: Prediction mode ("sample generation" or "mean estimation").
        batch_size: Batch size for prediction.
    Returns:
        An iterable of dictionaries, containing original controls and predicted perturbed samples."""
    all_incoming_rows = list(rows)  # Materialize rows for the current group
    if not all_incoming_rows:
        return iter([])

    control_rows_in_group = []
    perturbed_rows_in_group = []
    for row in all_incoming_rows:
        if row.get("perturbation_type") == "control":
            control_rows_in_group.append(row)
        else:
            perturbed_rows_in_group.append(row)

    # Process control rows to be yielded or skipped if classification-ready output requested.
    processed_control_rows = []
    if not classification_ready_output:
        for control_row in control_rows_in_group:
            processed_row = control_row.copy()
            if "counts" in processed_row:
                processed_row["counts_pred"] = processed_row.pop("counts")
            processed_row["modality"] = modality
            processed_control_rows.append(processed_row)

    # Case 1: No perturbed samples in this group.
    if not perturbed_rows_in_group:  # Only control samples
        return iter([])

    # Case 2: Perturbed samples exist, but no control samples for reference.
    if not control_rows_in_group:  # Only perturbed samples, no controls for reference
        # print(f"Warning: Group for study '{all_incoming_rows[0].get('study', 'Unknown')}' contains {len(perturbed_rows_in_group)} perturbed samples but no control samples for reference. Skipping these perturbed samples.")
        # Optionally, if perturbed samples should be passed through without prediction:
        # yield from perturbed_rows_in_group
        return iter([])

    # Case 3: Both controls and perturbed samples exist.
    # `to_ref_conditioned_pert` expects an iterable of all rows for the group.
    # It will internally separate controls and perturbed to create conditioned perturbed rows.
    conditioned_perturbed_rows_for_prediction = to_ref_conditioned_pert(
        iter(all_incoming_rows)
    )

    # Materialize to check if conditioning was successful and to reuse the iterator.
    materialized_conditioned_rows = list(conditioned_perturbed_rows_for_prediction)

    if not materialized_conditioned_rows:
        return

    # Predict on the successfully conditioned perturbed rows.
    predicted_perturbed_samples_iter = (
        predict_stream(  # This stream contains metadata, counts_pred, and latents
            rows=iter(materialized_conditioned_rows),
            modality=modality,
            model=model,
            batch_size=batch_size,
            mode=mode,
            conditioning=("technical", "biological"),
            total_count=total_count,
            fixed_total_count=fixed_total_count,
            deterministic_latents=deterministic_latents,
        )
    )

    # Define a helper to remove latent keys from prediction outputs
    def remove_latent_keys(row_dict):
        return {k: v for k, v in row_dict.items() if not k.endswith("_latent")}

    def rename_counts(row_dict: dict[str, Any]) -> dict[str, Any]:
        rd = remove_latent_keys(row_dict)
        if classification_ready_output and "counts_pred" in rd:
            rd = rd.copy()
            rd["counts"] = rd.pop("counts_pred")
        return rd

    processed_predicted_perturbed_samples_iter = pipe(
        predicted_perturbed_samples_iter, map(rename_counts)
    )

    # First, yield all original control rows (with 'counts' renamed)
    if not classification_ready_output:
        yield from processed_control_rows
        yield from processed_predicted_perturbed_samples_iter
    else:
        # Classification-ready: only perturbed rows with counts
        yield from processed_predicted_perturbed_samples_iter


def predict_perturbation_ref_study_fast(
    rows: Iterable[dict[str, Any]],
    model: SynthesizeBioModel,
    modality: str,
    mode: str,
    batch_size: int = BATCH_SIZE,
    total_count: int = 10_000_000,
    fixed_total_count: bool = False,
    classification_ready_output: bool = False,
    deterministic_latents: bool = False,
) -> Iterable[dict[str, Any]]:
    """
    Fast path for reference-conditioned perturbation prediction with pre-attached
    references, avoiding pandas joins/materialization.

    Expectations:
    - Each synthetic perturbed row has 'counts_ref' populated from its source control
      (see stream_utils where we copy a control's counts into counts_ref and drop counts).
    - Technical/biological metadata already match the reference control context.

    Behavior:
    - Filters out control rows and predicts only perturbed rows.
    - Uses conditioning (technical+biological) together with counts_ref to enforce 1:1
      reference conditioning without any pairing hint.
    - When classification_ready_output=True, emits rows with 'counts' (renamed) and no latents.
    """

    def _filtered_iter(it):
        for r in it:
            if r.get("perturbation_type") == "control":
                continue
            rr = dict(r)
            rr.pop("ref_experiment_accession_hint", None)
            yield rr

    preds_iter = predict_stream(
        rows=_filtered_iter(rows),
        modality=modality,
        model=model,
        batch_size=batch_size,
        mode=mode,
        conditioning=("technical", "biological"),
        total_count=total_count,
        fixed_total_count=fixed_total_count,
        deterministic_latents=deterministic_latents,
    )

    def remove_latent_keys(row_dict):
        return {k: v for k, v in row_dict.items() if not k.endswith("_latent")}

    def rename_counts(row_dict: dict[str, Any]) -> dict[str, Any]:
        rd = remove_latent_keys(row_dict)
        if classification_ready_output and "counts_pred" in rd:
            rd = rd.copy()
            rd["counts"] = rd.pop("counts_pred")
        return rd

    return pipe(preds_iter, map(rename_counts))


def substitute_controls(
    model: SynthesizeBioModel,
    modality: str,
    batch_size: int,
    mode: str,
    rows_for_group: Iterable[dict[str, Any]],
    total_count: int = 10_000_000,
    fixed_total_count: bool = False,
) -> Iterable[dict[str, Any]]:
    """
    Substitutes actual control samples in a group with synthetically generated ones.

    For a given group of samples (assumed to be from the same technical group/study):
    1. Separates control and perturbed samples.
    2. If controls exist, it uses `predict_common_technical` to generate synthetic
       versions of these controls. The 'counts' field of these synthetic controls
       will be their predicted counts.
    3. Yields these synthetic controls.
    4. Yields the original perturbed samples unchanged.
    This is used in the "fully synthetic" perturbation prediction pipeline.

    Args:
        model: The SynthesizeBioModel instance.
        modality: Data modality.
        batch_size: Batch size for prediction.
        mode: Prediction mode for generating synthetic controls.
        rows_for_group: An iterable of dictionaries for samples in the current group."""
    control_rows_buffer = []
    perturbed_rows_buffer = []
    has_any_rows = False

    for row in rows_for_group:
        assert "counts_ref" not in row, (
            "Input rows should not have 'counts_ref' for fully synthetic."
        )

        has_any_rows = True
        if row.get("perturbation_type") == "control":
            control_rows_buffer.append(row)
        else:
            perturbed_rows_buffer.append(row)

    if not has_any_rows:
        return iter([])

    if not control_rows_buffer:  # No controls in this group
        # If no controls, just pass through perturbed rows.
        # predict_perturbation_ref_study might not be able to process them.
        # print(f"Warning: No control samples found in the current group. Perturbed samples will be passed through.")
        yield from perturbed_rows_buffer
        return

    # Use predict_common_technical to generate synthetic controls for the current group.
    # This ensures controls within this group share a common technical prediction basis.
    synthetic_control_predictions = list(
        predict_common_technical(
            study_data=iter(control_rows_buffer),
            model=model,
            modality=modality,
            batch_size=batch_size,
            mode=mode,
            total_count=total_count,
            fixed_total_count=fixed_total_count,
        )
    )
    if len(synthetic_control_predictions) != len(control_rows_buffer):
        raise RuntimeError(
            f"Expected {len(control_rows_buffer)} synthetic controls, "
            f"got {len(synthetic_control_predictions)}"
        )

    for orig_ctl, synth_ctl in zip(control_rows_buffer, synthetic_control_predictions):
        out = orig_ctl.copy()
        out["counts"] = synth_ctl["counts_pred"]
        yield out

    # Yield the original perturbed samples
    yield from perturbed_rows_buffer


def predict_perturbation_fully_synthetic(
    modality: str,
    input_path: str,
    output_path: str,
    mode: str,
    batch_size: int = BATCH_SIZE,
    output_batch_size: int = OUTPUT_BATCH_SIZE,
    strict: bool = True,
    total_count: int = 10_000_000,
    fixed_total_count: bool = False,
    replicates: int = 1,
) -> None:
    """
    Predicts perturbation effects using fully synthetic reference (control) data.

    This pipeline first replaces actual control samples within each technical group
    with synthetically generated controls (via `substitute_controls`). Then,
    `predict_perturbation_ref_study` is used to predict outcomes for perturbed
    samples, using these synthetic controls as references.

    Args:
        modality: Data modality.
        input_path: Path to input Parquet files.
        mode: Prediction mode for both control generation and final prediction.
        output_path: Path for output Parquet dataset.
        batch_size: Batch size for model prediction.
        output_batch_size: Batch size for writing output Parquet files."""
    model = load_model(modality=modality, strict=strict).eval()
    raw_data_stream = stream_parquets(input_path, replicates=replicates)

    # Define a key function for grouping by technical metadata columns
    technical_group_key = lambda row: tuple(
        row.get(k, "N/A_TECH_KEY") for k in data.GROUPED_COLUMNS["technical"]
    )

    processed_data_iter = pipe(
        raw_data_stream,
        curry(groupby, key=technical_group_key),
        map(
            lambda group_tuple: substitute_controls(
                model=model,
                modality=modality,
                batch_size=batch_size,
                mode=mode,
                rows_for_group=group_tuple[1],
                total_count=total_count,
                fixed_total_count=fixed_total_count,
            )
        ),
        map(
            lambda group_specific_rows_iter: predict_perturbation_ref_study(
                rows=list(
                    group_specific_rows_iter
                ),  # Materialize for predict_perturbation_ref_study
                model=model,
                modality=modality,
                mode=mode,
                batch_size=batch_size,
                total_count=total_count,
                fixed_total_count=fixed_total_count,
            )
        ),
        concat,
    )

    write_parquet(output_path, processed_data_iter, output_batch_size)
    print(f"Fully synthetic perturbation predictions written to {output_path}")
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run inference to predict expression from metadata."
    )
    parser.add_argument(
        "--modality",
        type=str,
        default="bulk",
        help="Modality of the data (e.g., 'bulk', 'czi').",
    )
    parser.add_argument(
        "--input_path",
        type=str,
        help="Path to the input parquets.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        help="Path to the output parquets.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=BATCH_SIZE, help="Batch size for prediction."
    )
    parser.add_argument(
        "--output_batch_size",
        type=int,
        default=OUTPUT_BATCH_SIZE,
        help="Batch size for writing output parquets.",
    )
    parser.add_argument(
        "--classification_ready_output",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If set with ref_predict_perturbation, drop controls and emit counts (renamed from counts_pred) ready for classification.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["mean estimation", "sample generation"],
        default="mean estimation",
        help="Prediction mode: 'mean estimation' or 'sample generation'.",
    )
    parser.add_argument(
        "--total_count",
        type=int,
        default=10_000_000,
        help="Total count for scaling predicted expression when fixed_total_count is True or no reference expression is available.",
    )
    parser.add_argument(
        "--fixed_total_count",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If True, always use the 'total_count' parameter for scaling. If False, use total counts from reference expression when available.",
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=[
            "predict_expression",
            "classify_samples",
            "ref_predict_perturbation",
            "predict_perturbation",
        ],
        default="predict_expression",
        help="Task to perform: predict expression, classify samples, predict perturbation with reference, or context transfer.",
    )
    parser.add_argument(
        "--use_common_technical",
        action=argparse.BooleanOptionalAction,  # Allows --use_common_technical or --no-use_common_technical
        default=False,
        help="Whether to use common technical factor prediction (default: True). Set to False to predict each sample independently.",
    )
    parser.add_argument(
        "--strict",
        action=argparse.BooleanOptionalAction,  # Allows --strict or --no-strict
        default=True,
        help="Use if loading model in non-strict mode.",
    )
    parser.add_argument(
        "--replicates",
        type=int,
        default=1,
        help="Number of predictions per input datapoint.",
    )
    args = parser.parse_args()

    if args.task == "predict_expression":
        parquet_to_parquet(
            modality=args.modality,
            input_path=args.input_path,
            output_path=args.output_path,
            mode=args.mode,
            use_common_technical=args.use_common_technical,
            batch_size=args.batch_size,
            output_batch_size=args.output_batch_size,
            strict=args.strict,
            total_count=args.total_count,
            fixed_total_count=args.fixed_total_count,
            replicates=args.replicates,
        )
    elif args.task == "classify_samples":
        classify_parquet(
            modality=args.modality,
            input_path=args.input_path,
            output_path=args.output_path,
            mode=args.mode,
            batch_size=args.batch_size,
            output_batch_size=args.output_batch_size,
            strict=args.strict,
            replicates=args.replicates,
        )
    elif args.task == "ref_predict_perturbation":
        # predicts perturbation using observed reference
        ref_parquet_to_parquet(
            modality=args.modality,
            input_path=args.input_path,
            output_path=args.output_path,
            mode=args.mode,
            batch_size=args.batch_size,
            output_batch_size=args.output_batch_size,
            strict=args.strict,
            total_count=args.total_count,
            fixed_total_count=args.fixed_total_count,
            replicates=args.replicates,
            classification_ready_output=args.classification_ready_output,
        )
    elif args.task == "predict_perturbation":
        # predicts perturbation using fully synthetic data
        predict_perturbation_fully_synthetic(
            modality=args.modality,
            input_path=args.input_path,
            output_path=args.output_path,
            batch_size=args.batch_size,
            output_batch_size=args.output_batch_size,
            strict=args.strict,
            mode=args.mode,
            total_count=args.total_count,
            fixed_total_count=args.fixed_total_count,
            replicates=args.replicates,
        )

    else:
        # This case should not be reachable due to argparse choices
        print(f"Error: Unknown task '{args.task}'")
        parser.print_help()
        exit(1)
