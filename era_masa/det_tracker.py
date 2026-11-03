"""Detector-mode MASA tracker with risk-aware fusion (MASA-RF).

Lifecycle follows ByteTrack/BoT-SORT conventions (high/low score split,
unconfirmed tracks, lost-track buffer scaled by FPS). Association fuses MASA
appearance with IoU and a scale-normalised centre score; appearance memory is
updated with an EMA whose rate shrinks with occlusion risk instead of being
frozen. Every component has a switch so ablations run the same code path.
"""
from dataclasses import dataclass, field, asdict
import numpy as np
from scipy.special import softmax
from .assignment import assign
from .geometry import centers, overlap, unit_vectors, xyah
from .kalman import BoxKalman


@dataclass
class RFConfig:
    high_thresh: float = 0.6
    low_thresh: float = 0.1
    new_track_thresh: float = 0.7
    track_buffer: int = 30
    match_thresh: float = 0.8          # accept stage-1 pairs with cost < match_thresh
    second_thresh: float = 0.5         # BYTE stage IoU distance threshold
    unconfirmed_thresh: float = 0.7
    appearance_weight: float = 0.5     # 0 -> geometry only, 1 -> appearance only
    center_weight: float = 0.5         # inside geometry: IoU vs centre score
    appearance_mode: str = "masa"      # masa=(cos+bisoftmax)/2, cosine=raw cosine
    adaptive_weight: float = 0.0       # raise appearance weight where geometry is ambiguous (0 = fixed)
    ambiguity_margin: float = 0.3
    geometry_gate: float = 0.05        # pairs with geometry score below this are invalid
    appearance_gate: float = 0.0       # cosine below this is invalid unless IoU >= iou_override
    iou_override: float = 0.5
    age_penalty: float = 0.05
    age_appearance_gate: float = 1.0
    ema_alpha: float = 0.8             # weight of the new embedding (MASA convention)
    risk_memory: float = 0.9           # alpha *= max(0.1, 1 - risk_memory * occlusion_risk)
    freeze_risk: float = 0.0           # >0: hard freeze (alpha = 0) when occlusion risk >= this value
    fuse_score: bool = True            # ByteTrack fuse_score on similarity
    byte_stage: bool = True
    cmc: bool = True
    output_kf_box: bool = False

    def to_dict(self):
        return asdict(self)


@dataclass
class _Track:
    tid: int
    kf: BoxKalman
    emb: np.ndarray
    box: np.ndarray
    last_frame: int
    activated: bool
    state: str = "tracked"  # tracked | lost
    hits: int = 1


def _apply_affine(kf, H):
    """Warp an xyah Kalman state by a 2x3 similarity/affine camera motion."""
    R, t = H[:, :2], H[:, 2]
    s = float(np.sqrt(abs(np.linalg.det(R)))) or 1.0
    J = np.eye(8)
    J[0:2, 0:2] = R
    J[4:6, 4:6] = R
    J[3, 3] = s
    J[7, 7] = s
    kf.mean = J @ kf.mean
    kf.mean[:2] += t
    kf.covariance = J @ kf.covariance @ J.T


class RiskFusionTracker:
    def __init__(self, config=None, fps=30.0):
        self.cfg = config or RFConfig()
        self.max_lost = max(1, int(round(fps / 30.0 * self.cfg.track_buffer)))
        self.tracks = []
        self.next_id = 1
        self.frame = 0
        self.last_stats = {}

    # ---- association scores -------------------------------------------------
    def _appearance(self, det_emb, trk_emb):
        cos = np.clip(unit_vectors(det_emb) @ unit_vectors(trk_emb).T, -1, 1)
        if self.cfg.appearance_mode == "cosine":
            return cos
        raw = det_emb @ trk_emb.T
        return (cos + (softmax(raw, axis=0) + softmax(raw, axis=1)) / 2) / 2

    def _geometry(self, boxes, pred):
        iou = overlap(boxes, pred)
        scale = np.maximum(pred[:, 2:] - pred[:, :2], 1)
        delta = (centers(boxes)[:, None] - centers(pred)[None]) / scale[None]
        center = np.exp(-2 * np.sum(delta ** 2, axis=2))
        return iou, (1 - self.cfg.center_weight) * iou + self.cfg.center_weight * center

    def _fused_cost(self, boxes, scores, embs, tracks):
        cfg = self.cfg
        pred = np.stack([t.kf.box for t in tracks])
        iou, geo = self._geometry(boxes, pred)
        cos = np.clip(unit_vectors(embs) @ unit_vectors(np.stack([t.emb for t in tracks])).T, -1, 1)
        app = self._appearance(embs, np.stack([t.emb for t in tracks]))
        weight = cfg.appearance_weight
        if cfg.adaptive_weight and geo.size:
            # geometric ambiguity: small margin between best and second-best geometry candidate,
            # checked in both directions (detection row and track column)
            def margin(g, axis):
                if g.shape[axis] < 2:
                    return np.ones(g.shape[1 - axis])
                srt = np.sort(g, axis=axis)
                return (srt.take(-1, axis=axis) - srt.take(-2, axis=axis))
            m = np.minimum(margin(geo, 1)[:, None], margin(geo, 0)[None, :])
            ambiguity = 1 - np.clip(m / cfg.ambiguity_margin, 0, 1)
            weight = weight + (1 - weight) * cfg.adaptive_weight * ambiguity
        sim = weight * app + (1 - weight) * geo
        if cfg.fuse_score:
            sim = sim * scores[:, None]
        cost = 1 - sim
        gaps = self.frame - np.array([t.last_frame for t in tracks])
        support = np.clip((cos - 0.5) / 0.4, 0, 1)
        cost = cost + np.minimum(cfg.age_penalty * (gaps - 1), 0.25)[None] * (1 - cfg.age_appearance_gate * support)
        valid = geo >= cfg.geometry_gate
        if cfg.appearance_gate:
            valid &= (cos >= cfg.appearance_gate) | (iou >= cfg.iou_override)
        return cost, valid, iou

    # ---- main step ------------------------------------------------------------
    def update(self, boxes, scores, embs, warp=None):
        """boxes Nx4 xyxy, scores N, embs NxD for ONE frame. Returns Kx5 [x1,y1,x2,y2,id]."""
        cfg = self.cfg
        self.frame += 1
        boxes, scores, embs = np.asarray(boxes, float).reshape(-1, 4), np.asarray(scores, float), np.asarray(embs, float)
        keep = scores > cfg.low_thresh
        boxes, scores, embs = boxes[keep], scores[keep], embs[keep]
        high = scores > cfg.high_thresh
        hb, hs, he = boxes[high], scores[high], embs[high]
        lb, ls = boxes[~high], scores[~high]
        # occlusion risk of each high detection: largest fraction of it covered by another detection
        risk = np.zeros(len(hb))
        if len(hb) and len(boxes) > 1:
            ioa = overlap(hb, boxes, mode="ioa")
            ioa[np.arange(len(hb)), np.flatnonzero(high)] = 0  # drop self-overlap
            risk = ioa.max(axis=1)

        for t in self.tracks:
            t.kf.predict()
            if cfg.cmc and warp is not None:
                _apply_affine(t.kf, np.asarray(warp, float))

        st = {"stage1_matches": 0, "byte_matches": 0, "unconfirmed_matches": 0, "new_tracks": 0,
              "high_dets": int(len(hb)), "low_dets": int(len(lb)), "mean_risk": float(risk.mean()) if len(risk) else 0.0}
        confirmed = [t for t in self.tracks if t.activated]
        unconfirmed = [t for t in self.tracks if not t.activated]
        matched_t, out = set(), []
        det_free = np.ones(len(hb), bool)

        def commit(t, box, emb=None, r=0.0):
            t.kf.update(xyah(box[None])[0])
            t.box, t.last_frame, t.state, t.hits = box.copy(), self.frame, "tracked", t.hits + 1
            if emb is not None and np.linalg.norm(emb) > 1e-12:
                alpha = cfg.ema_alpha * max(0.1, 1 - cfg.risk_memory * r)
                if cfg.freeze_risk and r >= cfg.freeze_risk:
                    alpha = 0.0
                t.emb = (1 - alpha) * t.emb + alpha * emb
            matched_t.add(id(t))

        # stage 1: confirmed (tracked + lost) vs high detections, fused cost
        if confirmed and len(hb):
            cost, valid, _ = self._fused_cost(hb, hs, he, confirmed)
            for d, k in assign(cost, valid, cfg.match_thresh):
                commit(confirmed[k], hb[d], he[d], risk[d])
                det_free[d] = False
                st["stage1_matches"] += 1
        # stage 2 (BYTE): still-unmatched *tracked* tracks vs low detections, IoU only
        if cfg.byte_stage and len(lb):
            rest = [t for t in confirmed if id(t) not in matched_t and t.state == "tracked"]
            if rest:
                iou = overlap(lb, np.stack([t.kf.box for t in rest]))
                for d, k in assign(1 - iou, np.ones_like(iou, bool), cfg.second_thresh):
                    commit(rest[k], lb[d])  # low-score boxes never write memory
                    st["byte_matches"] += 1
        # stage 3: unconfirmed vs remaining high detections
        free = np.flatnonzero(det_free)
        if unconfirmed and len(free):
            cost, valid, _ = self._fused_cost(hb[free], hs[free], he[free], unconfirmed)
            for d, k in assign(cost, valid, cfg.unconfirmed_thresh):
                commit(unconfirmed[k], hb[free[d]], he[free[d]], risk[free[d]])
                unconfirmed[k].activated = True
                det_free[free[d]] = False
                st["unconfirmed_matches"] += 1
        # lifecycle
        survivors = []
        for t in self.tracks:
            if id(t) in matched_t:
                survivors.append(t)
            elif not t.activated:
                continue  # unconfirmed tracks die after one miss (ByteTrack)
            else:
                t.state = "lost"
                if self.frame - t.last_frame <= self.max_lost:
                    survivors.append(t)
        for d in np.flatnonzero(det_free):
            if hs[d] < cfg.new_track_thresh:
                continue
            t = _Track(self.next_id, BoxKalman(xyah(hb[d:d + 1])[0]), he[d].copy(), hb[d].copy(),
                       self.frame, activated=self.frame == 1)
            self.next_id += 1
            st["new_tracks"] += 1
            survivors.append(t)
            matched_t.add(id(t))
        self.tracks = survivors
        st["tracked"] = sum(t.state == "tracked" for t in survivors)
        st["lost"] = sum(t.state == "lost" for t in survivors)
        self.last_stats = st
        for t in self.tracks:
            if t.activated and t.last_frame == self.frame:
                box = t.kf.box if cfg.output_kf_box else t.box
                out.append([*box, t.tid])
        return np.array(out, float).reshape(-1, 5)
