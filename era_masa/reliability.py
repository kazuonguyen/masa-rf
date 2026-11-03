from dataclasses import dataclass
import numpy as np
from .geometry import centers, crowding, unit_vectors


@dataclass
class Reliability:
    risk: np.ndarray
    reliability: np.ndarray
    signals: dict
    overlap_iou: np.ndarray
    overlap_ioa: np.ndarray


def estimate(boxes, embeddings, predicted_boxes, clean_memory, memory_valid, motion, config):
    """Pairwise hand-crafted ECR. No GT identities/visibility enter this function."""
    n, m = len(boxes), len(predicted_boxes)
    cosine = np.clip(unit_vectors(embeddings) @ unit_vectors(clean_memory).T, -1, 1)
    temporal = np.clip((1 - cosine) / 2, 0, 1)
    # A competing track must be spatially local. Top-two avoids an NxMxM tensor.
    distances = np.linalg.norm(centers(boxes)[:, None] - centers(predicted_boxes)[None], axis=2)
    scale = np.maximum(boxes[:, 3] - boxes[:, 1], 1)[:, None]
    local = (distances / scale <= config.neighbor_radius) & memory_valid[None, :]
    candidates = np.where(local, cosine, -np.inf)
    if m:
        best_index = np.argmax(candidates, axis=1)
        best = candidates[np.arange(n), best_index]
        candidates[np.arange(n), best_index] = -np.inf
        second = candidates.max(axis=1)
        rivals = np.broadcast_to(best[:, None], (n, m)).copy()
        rivals[np.arange(n), best_index] = second
    else:
        rivals = np.empty((n, 0))
    rivals = np.where(np.isfinite(rivals), rivals, -1)
    rival = np.clip((rivals - cosine + config.rival_margin) / config.rival_margin, 0, 1)
    rival *= np.clip(rivals, 0, 1)
    iou, ioa = crowding(boxes)
    geometry = np.broadcast_to(np.maximum(iou, ioa)[:, None], (n, m))
    signals = {"temporal": temporal, "rival": rival,
               "geometry": geometry, "motion": np.clip(motion / config.motion_scale, 0, 1)}
    active = {
        "soft-iou": (), "temporal": ("temporal",),
        "temporal-rival": ("temporal", "rival"),
    }.get(config.variant, ("temporal", "rival", "geometry", "motion"))
    active = tuple(k for k in active if k != "motion" or config.motion_enabled)
    if config.variant == "soft-iou":
        risk = np.broadcast_to(iou[:, None], (n, m)).copy()
    else:
        total = sum(config.weights[k] for k in active)
        if total <= 0:
            raise ValueError("Active ECR signals must have positive total weight")
        risk = sum(config.weights[k] * signals[k] for k in active) / total
    reliability = np.clip(1 - risk, 0, 1)
    embedding_valid = np.linalg.norm(embeddings, axis=1) > 1e-12
    reliability *= embedding_valid[:, None] & memory_valid[None, :]
    return Reliability(1 - reliability, reliability, signals, iou, ioa)
