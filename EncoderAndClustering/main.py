from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Sequence

from EncoderAndClustering.encoding import encode
from EncoderAndClustering import time_series_clustering
from EncoderAndClustering.dico_departements import departements
import datetime as dt

def _load_dates(values: Sequence[str]) -> List[str]:
    """Return the provided sequence as a list of trimmed date strings."""
    return [value.strip() for value in values if value.strip()]


def _load_departments(values: Sequence[str]) -> List[str]:
    """Return a cleaned list of department identifiers."""
    return [value.strip() for value in values if value.strip()]


def _validate_dates(train_dates: Iterable[str], all_dates: Sequence[str]) -> None:
    """Ensure that every training date exists in the list of all available dates."""
    missing_dates = [date for date in train_dates if date not in all_dates]
    if missing_dates:
        raise ValueError(
            "The following training dates are missing from --all-dates: "
            + ", ".join(sorted(missing_dates))
        )


def _override_reader(target_bin_dir: Path) -> None:
    """Override the default reader used in the clustering module.

    The original implementation always loads data from the directory named
    ``path_to_target``.  This helper rewrites the module-level ``read_object``
    so that the directory provided via ``--target-bin-dir`` is honoured.
    """

    original_reader = time_series_clustering.read_object

    def _patched_reader(filename: str, _: Path) -> object:
        return original_reader(filename, target_bin_dir)

    time_series_clustering.read_object = _patched_reader

def find_dates_between(start, end):
    start_date = dt.datetime.strptime(start, '%Y-%m-%d').date()
    end_date = dt.datetime.strptime(end, '%Y-%m-%d').date()

    delta = dt.timedelta(days=1)
    date = start_date
    res = []
    while date < end_date:
            res.append(date.strftime("%Y-%m-%d"))
            date += delta
    return res

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create CatBoost encodings and cluster department level target time "
            "series."
        )
    )
    parser.add_argument(
        "--target-dir",
        type=Path,
        required=True,
        help="Directory containing the target tensors consumed by `encode`.",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        required=True,
        help="Label identifying the experiment (passed as `expe`).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where the encoder artefacts should be written.",
    )
    parser.add_argument(
        "--target-bin-dir",
        type=Path,
        required=True,
        help=(
            "Directory containing the per-department target pickles (used by "
            "time-series clustering)."
        ),
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=4,
        help="Number of clusters for the time-series clustering step.",
    )
    parser.add_argument(
        "--distance-metric",
        type=str,
        default="dtw",
        help="Distance metric passed to TimeSeriesKMeans.",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=10,
        help="Maximum number of iterations for the clustering algorithm.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed used by the clustering algorithm.",
    )
    parser.add_argument(
        "--cluster-output",
        type=Path,
        help="Optional path to save the node-to-cluster mapping as JSON.",
    )

    args = parser.parse_args()

    target_dir = args.target_dir
    experiment = args.experiment
    train_dates = find_dates_between('2017-06-12', '2020-12-31') + find_dates_between('2022-01-01', '2022-12-31')
    all_dates = find_dates_between('2017-06-12', '2024-12-31')
    train_departments = ['departement-'+dept for dept in departements]
    cluster_departments = train_departments
    drop_departments = []
    output_dir = Path(args.output_dir)
    target_bin_dir = Path(args.target_bin_dir)

    print("Starting CatBoost encoding generation...", flush=True)
    encode(
        path_to_target=target_dir,
        trainDates=train_dates,
        expe=experiment,
        train_departements=train_departments,
        dir_output=output_dir,
    )

    print("Encoding completed. Starting time-series clustering...", flush=True)
    _override_reader(target_bin_dir)

    cluster_model, node_cluster, all_clusters = (
        time_series_clustering.cluster_time_series_from_targets(
            departments=cluster_departments,
            train_dates=train_dates,
            all_dates=all_dates,
            drop_departments=drop_departments,
            n_clusters=args.n_clusters,
            distance_metric=args.distance_metric,
            max_iter=args.max_iter,
            random_state=args.random_state,
        )
    )

    print(
        f"Clustering finished. Fitted model with {args.n_clusters} clusters.",
        flush=True,
    )

    if args.cluster_output:
        args.cluster_output.parent.mkdir(parents=True, exist_ok=True)
        with args.cluster_output.open("w", encoding="utf-8") as fp:
            json.dump(node_cluster, fp, indent=2, sort_keys=True)
        print(f"Cluster assignments saved to {args.cluster_output}.")

    # Silence linter complaints about unused objects when the JSON output is omitted.
    _ = cluster_model, all_clusters

if __name__ == "__main__":
    main()
