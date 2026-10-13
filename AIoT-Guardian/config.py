

from __future__ import annotations
import os
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# ======================================================================
# PATHS
# ======================================================================
PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR   = Path(os.getenv("AIOT_DATA_DIR", PROJECT_ROOT / "datasets"))
OUTPUT_DIR = Path(os.getenv("AIOT_OUTPUT_DIR", PROJECT_ROOT / "AIoT_Guardian_Results"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = {
    "BETH":         DATA_DIR / "BETH.csv",
    "BoT-IoT":      DATA_DIR / "BoT-IoT.csv",
    "Edge-IIoTset": DATA_DIR / "Edge-IIoTset.csv",
}

# ======================================================================
# SPLITTING
# ======================================================================
RANDOM_STATE   = 42
TRAIN_FRACTION = 0.70
VAL_FRACTION   = 0.15
TEST_FRACTION  = 0.15

# ======================================================================
# mRMR
# ======================================================================
MRMR_TOP_K = 20

# ======================================================================
# DPSTS  (Dual ProbSparse Spatial-Temporal attention)
# ======================================================================
DPSTS_HEADS      = 2
DPSTS_D_MODEL    = 32
DPSTS_PROBSPARSE = 0.25
DPSTS_SEQ_LEN    = 4

# ======================================================================
# REFN detector + D-NRO optimizer
# ======================================================================
REFN_HIDDEN      = 64
REFN_EPOCHS      = 50
DNRO_LR_INIT     = 0.05
DNRO_BETA        = 0.9
DNRO_DREAM_NOISE = 0.02

# ======================================================================
# FLSM fuzzy severity thresholds
# ======================================================================
FLSM_LOW_MAX = 0.4
FLSM_MED_MAX = 0.7

# ======================================================================
# DDPG-MARL remediation
# ======================================================================
MARL_EPISODES = 20
MARL_AGENTS   = 3
MARL_ACTIONS  = [
    "isolate_device",
    "reroute_traffic",
    "reconfigure_firewall",
    "restart_service",
    "no_action",
]

# ======================================================================
# PSO rerouting
# ======================================================================
PSO_PARTICLES  = 10
PSO_ITERATIONS = 15

# ======================================================================
# SPEED GUARDRAILS
# ======================================================================
MAX_ROWS_FOR_MRMR        = 20_000
MAX_ROWS_FOR_DPSTS       = 20_000
MAX_ROWS_FOR_SHAP        = 100
MAX_ATTACKS_TO_REMEDIATE = 50

# ======================================================================
# SHARED RNG
# ======================================================================
RANDOM = np.random.default_rng(RANDOM_STATE)

# ======================================================================
# COLUMN HEURISTICS
# ======================================================================
TARGET_CANDIDATES = [
    "target", "label", "attack_label", "attack", "class", "category",
    "attack_type", "attack_cat", "is_attack", "sus", "evil",
]

NORMAL_LABELS = {
    "normal", "benign", "legitimate", "legit", "clean", "nonattack",
    "non_attack", "non-attack", "background", "0", "false", "no",
}

IDENTIFIER_COLUMNS = {
    "id", "flow_id", "uid", "uuid", "timestamp", "time_stamp",
    "src_ip", "source_ip", "dst_ip", "dest_ip", "destination_ip",
    "src_mac", "dst_mac", "source_mac", "destination_mac",
}

SOURCE_COL = "__source_dataset__"
TARGET_COL = "__target__"


# ======================================================================
# UTILITIES
# ======================================================================
def canonical_name(name) -> str:
    """Lower-case, strip punctuation -> snake_case token."""
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


def find_target_column(df: pd.DataFrame, dataset_name: str) -> str:
    """Locate the label column, preferring exact canonical matches."""
    lookup = {canonical_name(c): c for c in df.columns}

    for cand in TARGET_CANDIDATES:
        key = canonical_name(cand)
        if key in lookup:
            return lookup[key]

    for col in df.columns:
        name = canonical_name(col)
        if any(t in name for t in ("target", "label", "class",
                                   "attack_type", "is_attack")):
            return col

    raise ValueError(
        f"No target column found in {dataset_name}. Columns: {list(df.columns)}"
    )


def convert_target_to_binary(series: pd.Series, dataset_name: str) -> pd.Series:
    """Map an arbitrary label column to {0 = normal, 1 = attack}."""
    if series.isna().any():
        raise ValueError(f"{dataset_name} has missing target labels.")

    numeric = pd.to_numeric(series, errors="coerce")

    if numeric.notna().all():
        uniq = set(numeric.unique())
        if uniq.issubset({0, 1}):
            return numeric.astype(int)
        warnings.warn(f"{dataset_name}: mapping 0->Normal, nonzero->Attack.")
        return (numeric != 0).astype(int)

    labels = (
        series.astype(str).str.strip().str.lower()
        .str.replace(r"[\s/]+", "_", regex=True)
        .str.replace(r"[^a-z0-9_-]", "", regex=True)
    )
    return (~labels.isin(NORMAL_LABELS)).astype(int)


def is_target_like_column(column) -> bool:
    """True if the column should be dropped to avoid label leakage."""
    name = canonical_name(column)
    if "label" in name or "target" in name:
        return True
    if name in {"attack_type", "attack_cat", "attack_category",
                "class_label", "is_attack"}:
        return True
    return name in {canonical_name(c) for c in TARGET_CANDIDATES}