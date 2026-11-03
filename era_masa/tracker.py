from dataclasses import dataclass, field
import numpy as np
from scipy.special import softmax
from .assignment import assign
from .config import BASELINES, TrackerConfig
from .geometry import centers, overlap, overlap_groups, unit_vectors, xyah
from .kalman import BoxKalman
from .reliability import estimate


@dataclass
class Track:
    track_id: int
    box: np.ndarray
    embedding: np.ndarray
    clean_embedding: np.ndarray
    label: int
    last_frame: int
    kf: BoxKalman
    gallery: list = field(default_factory=list)
    velocity: np.ndarray | None = None


class ReliabilityTracker:
    def __init__(self, config=None):
        self.config = config or TrackerConfig()
        self.reset()

    def reset(self):
        self.tracks = {}
        self.next_id = 1
        self.last_frame = None
        self.embedding_dim = None
        self.diagnostics = []
        self.pair_diagnostics = []
        self.statistics = {"matches": 0, "new_tracks": 0, "frozen_updates": 0}

    def update(self, frame, boxes, embeddings, scores=None, labels=None, diagnostics="matched", camera_translation=None):
        """Return IDs in the exact input detection order. Frames are 1-indexed.

        Missing frames advance prediction and age; no boxes are emitted for
        predictions. This keeps the fixed-ground-truth-box experiment intact.
        """
        cfg = self.config
        boxes, embeddings = np.asarray(boxes, dtype=float), np.asarray(embeddings, dtype=float)
        n = len(boxes)
        scores = np.ones(n) if scores is None else np.asarray(scores, dtype=float)
        labels = np.zeros(n, dtype=int) if labels is None else np.asarray(labels, dtype=int)
        if boxes.shape != (n, 4) or embeddings.ndim != 2 or embeddings.shape[0] != n:
            raise ValueError("Expected boxes Nx4, embeddings NxD")
        if scores.shape != (n,) or labels.shape != (n,):
            raise ValueError("Expected scores/labels of shape N")
        if not all(np.isfinite(a).all() for a in (boxes, embeddings, scores)):
            raise ValueError("Nonfinite boxes, scores or embeddings")
        if n and np.any(boxes[:, 2:] <= boxes[:, :2]):
            raise ValueError("Boxes must have positive width and height")
        if embeddings.shape[1] < 1:
            raise ValueError("Embedding dimension must be positive")
        if self.embedding_dim is not None and self.embedding_dim != embeddings.shape[1]:
            raise ValueError("Embedding dimension changed within a sequence")
        if frame < 1 or int(frame) != frame or (self.last_frame is not None and frame <= self.last_frame):
            raise ValueError("Frame numbers must be positive and strictly increasing")
        if diagnostics not in ("none", "matched", "all"):
            raise ValueError("diagnostics must be none, matched or all")
        self.embedding_dim = embeddings.shape[1]
        self.diagnostics, self.pair_diagnostics = [], []
        self.statistics = {"matches": 0, "new_tracks": 0, "frozen_updates": 0}
        for tid in list(self.tracks):
            if frame - self.tracks[tid].last_frame > cfg.max_age:
                del self.tracks[tid]
        if cfg.camera_motion:
            if camera_translation is None:
                raise ValueError("Camera compensation enabled without measured image translation")
            translation = np.asarray(camera_translation, dtype=float)
            if translation.shape != (2,) or not np.isfinite(translation).all():
                raise ValueError("Camera translation must be finite shape (2,)")
            for trk in self.tracks.values():
                trk.box += np.tile(translation, 2)
                trk.kf.mean[:2] += translation
        use_kf = cfg.variant not in ("masa", "masa-hung")
        steps = frame - self.last_frame if self.last_frame is not None else 0
        if use_kf:
            for trk in self.tracks.values():
                for _ in range(steps):
                    trk.kf.predict()
        self.last_frame = int(frame)
        ids = np.full(n, -1, dtype=int)
        if not n:
            return ids
        # Stable score order matches MASA's input order for greedy association.
        order = np.argsort(-scores, kind="stable")
        boxes, embeddings, scores, labels = boxes[order], embeddings[order], scores[order], labels[order]
        tracks = list(self.tracks.values())
        m = len(tracks)
        matched = {}
        if m:
            predicted = np.stack([t.kf.box if use_kf else t.box for t in tracks])
            if cfg.variant == "observation":
                predicted = np.stack([t.box + (np.zeros(4) if t.velocity is None else t.velocity) *
                                      (frame - t.last_frame) for t in tracks])
            if cfg.variant == "robust-fusion":
                observed_prediction = np.stack([t.box + (np.zeros(4) if t.velocity is None else t.velocity) *
                                      (frame - t.last_frame) for t in tracks])
                predicted = cfg.observation_mix * observed_prediction + (1 - cfg.observation_mix) * predicted
            memory = np.stack([t.embedding for t in tracks])
            clean = np.stack([t.clean_embedding for t in tracks])
            measurements = xyah(boxes)
            motion = np.column_stack([t.kf.distance(measurements) for t in tracks])
            raw_similarity = embeddings @ memory.T
            cosine = np.clip(unit_vectors(embeddings) @ unit_vectors(memory).T, -1, 1)
            if cfg.variant == "gallery":
                for t, trk in enumerate(tracks):
                    if trk.gallery:
                        cosine[:, t] = np.maximum(cosine[:, t],
                            (unit_vectors(embeddings) @ unit_vectors(np.stack(trk.gallery)).T).max(axis=1))
            appearance = (cosine + (softmax(raw_similarity, axis=0) + softmax(raw_similarity, axis=1)) / 2) / 2
            # Paper/MASA distance mask, applied equally to all variants.
            distance_mask = np.ones((n, m))
            if cfg.max_distance != -1:
                distances = np.linalg.norm(centers(boxes)[:, None] - centers(predicted)[None], axis=2)
                gaps = frame - np.array([t.last_frame for t in tracks])
                radius = cfg.max_distance * np.exp(np.minimum(gaps / (cfg.fps / 6), 50))
                distance_mask = np.exp(-np.maximum(distances - radius[None], 0) / (100 / cfg.fps))
                appearance *= distance_mask
            valid = np.broadcast_to(scores[:, None] > cfg.object_score_threshold, (n, m)).copy()
            if cfg.with_categories:
                valid &= labels[:, None] == np.array([t.label for t in tracks])[None]
            iou = overlap(boxes, predicted)
            rel = estimate(boxes, embeddings, predicted, clean,
                           np.linalg.norm(clean, axis=1) > 1e-12, motion, cfg)
            cost = 1 - appearance
            if cfg.variant in BASELINES or cfg.variant == "memory-only":
                matched = self._paper_matches(boxes, cost, iou, cosine, valid, rel.overlap_iou)
            elif cfg.variant in ("hybrid", "cascade", "gallery", "observation", "robust-fusion"):
                motion_cost = np.clip(motion / cfg.motion_scale, 0, 1)
                blend = cfg.geometry_motion_mix if cfg.motion_enabled else 0
                geometry_cost = (1 - blend) * (1 - iou) + blend * motion_cost
                weight = cfg.appearance_weight
                if cfg.variant == "robust-fusion":
                    scale = np.maximum(predicted[:, 2:] - predicted[:, :2], 1)
                    delta = (centers(boxes)[:, None] - centers(predicted)[None]) / scale[None]
                    center_score = np.exp(-2 * np.sum(delta ** 2, axis=2))
                    geometry_score = (1 - cfg.center_weight) * iou + cfg.center_weight * center_score
                    geometry_cost = 1 - geometry_score
                    if cfg.adaptive_appearance:
                        sorted_scores = np.sort(geometry_score, axis=1)
                        margin = sorted_scores[:, -1] - (sorted_scores[:, -2] if m > 1 else 0)
                        ambiguity = 1 - np.clip(margin / 0.3, 0, 1)
                        weight = weight + (1 - weight) * cfg.adaptive_appearance * ambiguity[:, None]
                cost = weight * (1 - appearance) + (1 - weight) * geometry_cost
                if cfg.direction_weight:
                    for t, trk in enumerate(tracks):
                        if trk.velocity is not None:
                            delta = centers(boxes) - centers(trk.box[None])[0]
                            velocity = centers((trk.box + trk.velocity)[None])[0] - centers(trk.box[None])[0]
                            speed = np.linalg.norm(velocity)
                            if speed > 1e-6:
                                direction = np.clip(delta @ velocity / np.maximum(np.linalg.norm(delta, axis=1) * speed, 1e-6), -1, 1)
                                cost[:, t] = (1 - cfg.direction_weight) * cost[:, t] + cfg.direction_weight * (1 - direction) / 2
                cost = 1 - (1 - cost) * distance_mask
                if cfg.variant == "robust-fusion":
                    gaps = frame - np.array([t.last_frame for t in tracks])
                    memory_support = np.clip((cosine - 0.5) / 0.4, 0, 1)
                    cost += np.minimum(cfg.age_penalty * (gaps - 1), 0.25)[None] * (
                        1 - cfg.age_appearance_gate * memory_support)
                if cfg.variant == "cascade":
                    first = assign(1 - iou, valid, 1 - cfg.iou_threshold)
                    for d, t in first:
                        matched[d] = (t, cfg.ema_alpha if rel.overlap_iou[d] < cfg.overlap_low else cfg.fallback_alpha, "cascade-iou")
                    rows = [d for d in range(n) if d not in matched]
                    used = {x[0] for x in matched.values()}
                    cols = [t for t in range(m) if t not in used]
                    ix = np.ix_(rows, cols)
                    for d, t in assign(cost[ix], valid[ix], 1 - cfg.match_threshold):
                        matched[rows[d]] = (cols[t], cfg.ema_alpha, "cascade-hybrid")
                else:
                    for d, t in assign(cost, valid, 1 - cfg.match_threshold):
                        matched[d] = (t, cfg.ema_alpha, cfg.variant)
            else:
                motion_cost = np.clip(motion / cfg.motion_scale, 0, 1)
                blend = cfg.geometry_motion_mix if cfg.motion_enabled else 0.0
                geometric = (1 - blend) * (1 - iou) + blend * motion_cost
                cost = rel.reliability * (1 - appearance) + (1 - rel.reliability) * geometric
                # Spatial soft gating cannot be bypassed by the geometric branch.
                cost = 1 - (1 - cost) * distance_mask
                for d, t in assign(cost, valid, 1 - cfg.match_threshold):
                    matched[d] = (t, cfg.ema_alpha, "adaptive")
            if diagnostics == "all":
                for d in range(n):
                    for t in range(m):
                        self.pair_diagnostics.append(self._row(frame, int(order[d]), tracks[t].track_id,
                            d, t, rel, cost, 0, "candidate", valid[d, t], None))
            for d, (t, base_alpha, stage) in matched.items():
                trk = tracks[t]
                reliability = float(rel.reliability[d, t])
                decision_cost = 1 - iou if "iou" in stage else cost
                # Bidirectional margin: ambiguity can come from another detection
                # or another track. Compare only eligible edges, before assignment.
                row_other = [decision_cost[d, j] for j in range(m) if j != t and valid[d, j]]
                col_other = [decision_cost[i, t] for i in range(n) if i != d and valid[i, t]]
                alternative = min(row_other + col_other, default=1.0)
                margin = float(alternative - decision_cost[d, t])
                alpha = base_alpha
                if cfg.variant in ("era-full", "memory-only"):
                    if reliability < cfg.reliability_low or margin < cfg.margin_threshold:
                        alpha = 0.0
                    elif reliability >= cfg.reliability_high:
                        alpha = cfg.ema_alpha
                    else:
                        alpha = cfg.slow_alpha * (reliability - cfg.reliability_low) / (cfg.reliability_high - cfg.reliability_low)
                if cfg.variant == "robust-fusion":
                    alpha *= max(0.1, 1 - cfg.memory_risk_factor * (1 - reliability))
                # Never poison memory with an invalid embedding, in any variant.
                if np.linalg.norm(embeddings[d]) <= 1e-12:
                    alpha = 0.0
                self.statistics["matches"] += 1
                self.statistics["frozen_updates"] += int(alpha == 0)
                trk.embedding = (1 - alpha) * trk.embedding + alpha * embeddings[d]
                if alpha > 0 and cfg.variant == "gallery" and margin >= cfg.margin_threshold:
                    trk.gallery.append(embeddings[d].copy())
                    trk.gallery = trk.gallery[-cfg.gallery_size:]
                # Anchor evolves only after unambiguous, highly reliable matches.
                # Baselines do not use this anchor for association.
                if reliability >= cfg.reliability_high and margin >= cfg.margin_threshold and alpha > 0:
                    trk.clean_embedding = trk.embedding.copy()
                measured_velocity = (boxes[d] - trk.box) / max(frame - trk.last_frame, 1)
                if cfg.variant == "robust-fusion":
                    center_velocity = (measured_velocity[:2] + measured_velocity[2:]) / 2
                    measured_velocity = np.tile(center_velocity, 2)
                    trk.velocity = measured_velocity if trk.velocity is None else (
                        (1 - cfg.velocity_alpha) * trk.velocity + cfg.velocity_alpha * measured_velocity)
                else:
                    trk.velocity = measured_velocity
                trk.box, trk.label, trk.last_frame = boxes[d].copy(), int(labels[d]), int(frame)
                if use_kf:
                    trk.kf.update(measurements[d])
                ids[d] = trk.track_id
                if diagnostics != "none":
                    self.diagnostics.append(self._row(frame, int(order[d]), trk.track_id,
                        d, t, rel, decision_cost, alpha, stage, True, margin))
        for d in np.flatnonzero(ids < 0):
            if scores[d] <= cfg.init_score_threshold:
                continue
            tid = self.next_id
            self.next_id += 1
            self.tracks[tid] = Track(tid, boxes[d].copy(), embeddings[d].copy(), embeddings[d].copy(),
                                    int(labels[d]), int(frame), BoxKalman(xyah(boxes[d:d+1])[0]))
            ids[d] = tid
            self.statistics["new_tracks"] += 1
            if diagnostics != "none":
                self.diagnostics.append({"frame": int(frame), "detection_index": int(order[d]),
                    "track_id": tid, "stage": "new", "alpha": None, "margin": None,
                    "risk": None, "reliability": None})
        original_ids = np.empty(n, dtype=int)
        original_ids[order] = ids
        return original_ids

    def _paper_matches(self, boxes, cost, iou, cosine, valid, crowd):
        cfg = self.config
        variant = "masa-oa" if cfg.variant == "memory-only" else cfg.variant
        n, m = cost.shape
        result, used = {}, set()

        def stage(rows, metric, threshold, alpha, name, greedy=False):
            cols = [t for t in range(m) if t not in used]
            rows = [d for d in rows if d not in result]
            if not rows or not cols:
                return
            ix = np.ix_(rows, cols)
            for r, c in assign(metric[ix], valid[ix], threshold, greedy):
                d, t = rows[r], cols[c]
                result[d] = (t, alpha, name)
                used.add(t)

        if variant in ("masa", "masa-hung", "masa-hung-kf"):
            stage(range(n), cost, 1 - cfg.match_threshold, cfg.ema_alpha, "appearance", variant == "masa")
        else:
            stage(np.flatnonzero(crowd < cfg.overlap_low), cost, 1 - cfg.match_threshold,
                  cfg.ema_alpha, "isolated-appearance")
            if variant in ("masa-clust2", "masa-clust8"):
                for group in overlap_groups(boxes, cfg.overlap_low):
                    rows = [d for d in group if d not in result and crowd[d] <= cfg.overlap_high]
                    cols = [t for t in range(m) if t not in used]
                    candidates = [(cosine[d, t], d, t) for d in rows for t in cols
                                  if valid[d, t] and cost[d, t] < 1 - cfg.match_threshold]
                    if candidates:
                        _, d, t = max(candidates)
                        result[d] = (t, 0.2 if variant == "masa-clust2" else 0.8, "cluster-front")
                        used.add(t)
            stage(np.flatnonzero(crowd >= cfg.overlap_low), 1 - iou,
                  1 - cfg.iou_threshold, 0.0, "occluded-iou")
        if variant not in ("masa", "masa-hung"):
            stage(range(n), 1 - iou, 1 - cfg.iou_threshold, cfg.fallback_alpha, "fallback-iou")
        return result

    @staticmethod
    def _row(frame, detection_index, track_id, d, t, rel, cost, alpha, stage, valid, margin):
        return {"frame": int(frame), "detection_index": detection_index,
                "track_id": track_id, "stage": stage, "valid": bool(valid),
                "cost": float(cost[d, t]), "risk": float(rel.risk[d, t]),
                "reliability": float(rel.reliability[d, t]), "alpha": alpha, "margin": margin,
                "overlap_iou": float(rel.overlap_iou[d]), "overlap_ioa": float(rel.overlap_ioa[d]),
                **{k: float(v[d, t]) for k, v in rel.signals.items()}}
