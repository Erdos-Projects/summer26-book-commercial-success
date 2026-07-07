"""
Amazon Sales Rank history feature extraction.

Each ASIN's rank history lives in a JSON file named
``{ASIN}_com_norm.json`` under AMAZON_RANK_HISTORY_DIR.
The JSON maps Unix timestamp strings to integer rank values.
"""

import json
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from utils.paths import AMAZON_RANK_HISTORY_DIR


def load_rank_history(asin: str) -> dict[int, int]:
    """
    Load the full rank-history time-series for one ASIN.

    Args:
        asin: 10-character Amazon ASIN string.

    Outputs:
        None

    Returns:
        Dict mapping Unix timestamp (int) to rank (int), or an
        empty dict if the ASIN has no rank-history file.
    """
    path = os.path.join(AMAZON_RANK_HISTORY_DIR, f"{asin}_com_norm.json")
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)
    return {int(k): int(v) for k, v in raw.items()}


def compute_rank_features(history: dict[int, int]) -> dict:
    """
    Derive summary rank features from a single ASIN's rank history.

    Args:
        history: dict mapping Unix timestamp to rank value, as returned
                 by ``load_rank_history``.

    Outputs:
        None

    Returns:
        Dict with keys:
            rank_best         — minimum (best) rank observed
            rank_median       — median rank
            rank_latest       — rank at the most recent timestamp
            rank_observations — total number of rank snapshots
            rank_time_in_top1k  — fraction of snapshots with rank ≤ 1,000
            rank_time_in_top10k — fraction of snapshots with rank ≤ 10,000
    """
    if not history:
        return {
            "rank_best": np.nan,
            "rank_median": np.nan,
            "rank_latest": np.nan,
            "rank_observations": 0,
            "rank_time_in_top1k": np.nan,
            "rank_time_in_top10k": np.nan,
        }

    ranks = np.array(list(history.values()), dtype=float)
    latest_ts = max(history)

    return {
        "rank_best": float(ranks.min()),
        "rank_median": float(np.median(ranks)),
        "rank_latest": float(history[latest_ts]),
        "rank_observations": int(len(ranks)),
        "rank_time_in_top1k": float((ranks <= 1_000).mean()),
        "rank_time_in_top10k": float((ranks <= 10_000).mean()),
    }


def batch_compute_rank_features(asins: list[str], show_progress: bool = True) -> pd.DataFrame:
    """
    Compute rank features for a list of ASINs.

    Only ASINs that have a rank-history file contribute non-NaN rows;
    ASINs with no file still appear in the output with NaN feature values.

    Args:
        asins: list of ASIN strings to process.
        show_progress: whether to display a tqdm progress bar.

    Outputs:
        None

    Returns:
        DataFrame indexed by ASIN with one row per ASIN and one column
        per rank feature (rank_best, rank_median, rank_latest,
        rank_observations, rank_time_in_top1k, rank_time_in_top10k).
    """
    iterator = tqdm(asins, desc="Rank features") if show_progress else asins
    records = []
    for asin in iterator:
        history = load_rank_history(asin)
        feats = compute_rank_features(history)
        feats["asin"] = asin
        records.append(feats)

    return pd.DataFrame(records).set_index("asin")


def available_rank_asins() -> set[str]:
    """
    Return the set of ASINs that have a rank-history file on disk.

    Args:
        None

    Outputs:
        None

    Returns:
        Set of ASIN strings (10-character codes).
    """
    if not os.path.isdir(AMAZON_RANK_HISTORY_DIR):
        return set()
    return {
        fname.replace("_com_norm.json", "")
        for fname in os.listdir(AMAZON_RANK_HISTORY_DIR)
        if fname.endswith("_com_norm.json")
    }
