"""NumPy drop-in for cython_bbox.bbox_overlaps (same +1 pixel area convention)."""
import numpy as np


def bbox_overlaps(boxes, query_boxes):
    boxes = np.asarray(boxes, dtype=np.float64)
    query = np.asarray(query_boxes, dtype=np.float64)
    if boxes.size == 0 or query.size == 0:
        return np.zeros((len(boxes), len(query)), dtype=np.float64)
    q_area = (query[:, 2] - query[:, 0] + 1) * (query[:, 3] - query[:, 1] + 1)
    b_area = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
    iw = np.minimum(boxes[:, None, 2], query[None, :, 2]) - np.maximum(boxes[:, None, 0], query[None, :, 0]) + 1
    ih = np.minimum(boxes[:, None, 3], query[None, :, 3]) - np.maximum(boxes[:, None, 1], query[None, :, 1]) + 1
    inter = np.where((iw > 0) & (ih > 0), iw * ih, 0.0)
    return inter / (b_area[:, None] + q_area[None, :] - inter)
