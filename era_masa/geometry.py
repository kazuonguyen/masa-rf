import numpy as np


def unit_vectors(features):
    features = np.asarray(features, dtype=float)
    norms = np.linalg.norm(features, axis=-1, keepdims=True)
    return features / np.maximum(norms, 1e-12)


def overlap(a, b, mode="iou"):
    """NxM intersection-over-union or intersection-over-area of a."""
    a, b = np.asarray(a), np.asarray(b)
    lo = np.maximum(a[:, None, :2], b[None, :, :2])
    hi = np.minimum(a[:, None, 2:], b[None, :, 2:])
    intersection = np.prod(np.maximum(hi - lo, 0), axis=2)
    aa = np.prod(np.maximum(a[:, 2:] - a[:, :2], 0), axis=1)
    ba = np.prod(np.maximum(b[:, 2:] - b[:, :2], 0), axis=1)
    denom = aa[:, None] if mode == "ioa" else aa[:, None] + ba[None, :] - intersection
    return intersection / np.maximum(denom, 1e-12)


def crowding(boxes):
    n = len(boxes)
    if n == 0:
        return np.empty(0), np.empty(0)
    iou = overlap(boxes, boxes)
    ioa = overlap(boxes, boxes, "ioa")
    np.fill_diagonal(iou, 0)
    np.fill_diagonal(ioa, 0)
    return iou.max(axis=1), ioa.max(axis=1)


def centers(boxes):
    return (boxes[:, :2] + boxes[:, 2:]) / 2


def xyah(boxes):
    sizes = boxes[:, 2:] - boxes[:, :2]
    return np.column_stack((centers(boxes), sizes[:, 0] / sizes[:, 1], sizes[:, 1]))


def from_xyah(state):
    h = max(float(state[3]), 1e-3)
    w = max(float(state[2]) * h, 1e-3)
    size = np.array([w, h])
    return np.r_[state[:2] - size / 2, state[:2] + size / 2]


def overlap_groups(boxes, threshold):
    graph = overlap(boxes, boxes) >= threshold
    seen, groups = set(), []
    for i in range(len(boxes)):
        if i in seen:
            continue
        pending, group = [i], []
        seen.add(i)
        while pending:
            j = pending.pop()
            group.append(j)
            for k in np.flatnonzero(graph[j]):
                if int(k) not in seen:
                    seen.add(int(k))
                    pending.append(int(k))
        if len(group) > 1:
            groups.append(group)
    return groups
