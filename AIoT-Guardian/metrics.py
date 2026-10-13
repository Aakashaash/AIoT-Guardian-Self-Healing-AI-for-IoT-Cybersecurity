

from __future__ import annotations

import os

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


# ======================================================================
# METRIC SUITE
# ======================================================================
def calculate_metrics(y_true, y_pred, y_prob) -> dict:
    """Full binary-classification metric suite."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        auc = None

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_sensitivity": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) > 0 else None,
        "false_negative_rate": float(fn / (fn + tp)) if (fn + tp) > 0 else None,
        "detection_rate": float(tp / (tp + fn)) if (tp + fn) > 0 else None,
        "auc_roc": auc,
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }


# ======================================================================
# FDRVM
# ======================================================================
def fdrvm_validate(post_state: dict, pre_state: dict, threshold: float = 0.05):
    """
    Feedback-Driven Recovery Validation.

    Parameters
    ----------
    post_state, pre_state : dict with keys
        attack_prob  -- mean attack probability
        connectivity -- normalised connectivity in [0, 1]
        throughput   -- normalised throughput in [0, 1]

    Returns
    -------
    (success: bool, feedback: dict)
    """
    residual_attack = post_state["attack_prob"] - pre_state["attack_prob"]
    connectivity_ok = post_state["connectivity"] > 0.9 * pre_state["connectivity"]
    throughput_ok = post_state["throughput"] > 0.8 * pre_state["throughput"]

    success = (residual_attack < -threshold) and connectivity_ok and throughput_ok

    feedback = {
        "residual_attack_delta": float(residual_attack),
        "connectivity_ok": bool(connectivity_ok),
        "throughput_ok": bool(throughput_ok),
        "success": bool(success),
    }
    return success, feedback


# ======================================================================
# RESOURCE MONITOR
# ======================================================================
def resource_monitor() -> dict:
    """Best-effort CPU / RSS sampling; silently degrades if psutil absent."""
    out = {"cpu_percent": None, "memory_mb": None}
    try:
        import psutil

        proc = psutil.Process(os.getpid())
        out["cpu_percent"] = float(psutil.cpu_percent(interval=0.1))
        out["memory_mb"] = float(proc.memory_info().rss / (1024 * 1024))
    except Exception:
        pass
    return out