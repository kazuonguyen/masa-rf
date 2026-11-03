"""Adapters running *official* tracker source on cached detections/embeddings.

Only external inputs are replaced: images (unused by these trackers once
features and camera motion are injected), the ReID network (-> cached MASA
embeddings, L2-normalised) and GMC (-> cached sparse-optical-flow affine).
Each factory mutates sys.path, so run exactly one method per process.
"""
import sys
import types
from pathlib import Path
import numpy as np

# Upstream ByteTrack/BoT-SORT use aliases removed in NumPy 2; restore them in memory only.
for _alias, _type in (("float", float), ("int", int), ("bool", bool)):
    if _alias not in np.__dict__:
        setattr(np, _alias, _type)

ROOT = Path(__file__).resolve().parents[1]
TP = ROOT / "third_party"
SHIMS = Path(__file__).resolve().parent / "shims"

# Published default hyper-parameters of each method (MOT17/DanceTrack val settings).
DEFAULTS = {
    "bytetrack": dict(track_thresh=0.6, track_buffer=30, match_thresh=0.8),
    "ocsort": dict(det_thresh=0.6, max_age=30, min_hits=3, iou_threshold=0.3, delta_t=3, inertia=0.2, use_byte=False),
    "botsort": dict(track_high_thresh=0.6, track_low_thresh=0.1, new_track_thresh=0.7, track_buffer=30,
                    match_thresh=0.8, proximity_thresh=0.5, appearance_thresh=0.25, with_reid=False),
    "botsort-reid": dict(track_high_thresh=0.6, track_low_thresh=0.1, new_track_thresh=0.7, track_buffer=30,
                         match_thresh=0.8, proximity_thresh=0.5, appearance_thresh=0.25, with_reid=True),
    "deepocsort": dict(det_thresh=0.6, max_age=30, min_hits=3, iou_threshold=0.3, delta_t=3, inertia=0.2,
                       w_association_emb=0.75, alpha_fixed_emb=0.95, aw_param=0.5),
    "sort": dict(det_thresh=0.6, max_age=30, min_hits=3, iou_threshold=0.3),
}


def _norm(e):
    return e / np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-12)


class ByteTrackAdapter:
    def __init__(self, params, fps, mot20):
        sys.path[:0] = [str(SHIMS), str(TP / "ByteTrack")]
        from yolox.tracker.byte_tracker import BYTETracker
        args = types.SimpleNamespace(**{**DEFAULTS["bytetrack"], **params}, mot20=mot20)
        self.t = BYTETracker(args, frame_rate=fps)

    def update(self, boxes, scores, embs, warp, hw):
        dets = np.c_[boxes, scores] if len(boxes) else np.empty((0, 5))
        out = self.t.update(dets, list(hw), list(hw))
        return np.array([[*t.tlbr, t.track_id] for t in out]).reshape(-1, 5)


class OCSortAdapter:
    def __init__(self, params, fps, mot20, sort_only=False):
        sys.path[:0] = [str(SHIMS), str(TP / "OC_SORT")]
        from trackers.ocsort_tracker.ocsort import OCSort
        p = {**DEFAULTS["ocsort"], **params}
        if sort_only:  # SORT == OC-SORT with OCM/OCR/ORU disabled is not exact; keep inertia 0 as closest
            p.update(inertia=0.0)
        self.t = OCSort(**p)

    def update(self, boxes, scores, embs, warp, hw):
        dets = np.c_[boxes, scores] if len(boxes) else np.empty((0, 5))
        return self.t.update(dets, list(hw), list(hw)).reshape(-1, 5)


class BoTSortAdapter:
    def __init__(self, params, fps, mot20, reid):
        sys.path[:0] = [str(SHIMS), str(TP / "BoT-SORT")]
        from tracker.bot_sort import BoTSORT
        p = {**DEFAULTS["botsort-reid" if reid else "botsort"], **params}
        args = types.SimpleNamespace(**{**p, "with_reid": False}, mot20=mot20, cmc_method="none",
                                     name="cache", ablation=False, fast_reid_config=None,
                                     fast_reid_weights=None, device="cpu")
        self.t = BoTSORT(args, frame_rate=fps)
        args.with_reid = reid
        self.args = args
        self.feats = None
        self.warp = np.eye(2, 3)
        adapter = self
        self.t.encoder = types.SimpleNamespace(inference=lambda img, dets: adapter.feats)
        self.t.gmc = types.SimpleNamespace(apply=lambda img, dets: adapter.warp.copy())

    def update(self, boxes, scores, embs, warp, hw):
        a = self.args
        low = scores > a.track_low_thresh
        high = scores[low] > a.track_high_thresh
        self.feats = _norm(embs[low][high]) if len(embs) else np.empty((0, 256))
        self.warp = np.eye(2, 3) if warp is None else np.asarray(warp, float)
        dets = np.c_[boxes, scores] if len(boxes) else np.empty((0, 5))
        out = self.t.update(dets, None)
        return np.array([[*t.tlbr, t.track_id] for t in out]).reshape(-1, 5)


class DeepOCSortAdapter:
    def __init__(self, params, fps, mot20):
        sys.path[:0] = [str(SHIMS), str(TP / "Deep-OC-SORT")]
        for name in ("torchreid", "external", "external.adaptors", "external.adaptors.fastreid_adaptor"):
            sys.modules.setdefault(name, types.ModuleType(name))
        sys.modules["external.adaptors.fastreid_adaptor"].FastReID = object
        from trackers.integrated_ocsort_embedding import ocsort as docs
        adapter = self

        class Embedder:
            def __init__(self, *a, **k):
                pass

            def compute_embedding(self, img, bbox, tag):
                return adapter.feats

            def dump_cache(self):
                pass

        class CMC:
            def compute_affine(self, img, bbox, tag):
                return adapter.warp.copy()

            def dump_cache(self):
                pass

        docs.EmbeddingComputer = Embedder
        docs.CMCComputer = CMC
        p = {**DEFAULTS["deepocsort"], **params}
        self.det_thresh = p["det_thresh"]
        self.t = docs.OCSort(args=types.SimpleNamespace(dataset="cache", test_dataset=False), grid_off=True, **p)
        self.feats, self.warp = None, np.eye(2, 3)

    def update(self, boxes, scores, embs, warp, hw):
        keep = scores > self.det_thresh
        self.feats = _norm(embs[keep]) if keep.any() else np.empty((0, 256))
        self.warp = np.eye(2, 3) if warp is None else np.asarray(warp, float)
        dets = np.c_[boxes, scores] if len(boxes) else np.empty((0, 5))
        h, w = hw
        out = self.t.update(dets, types.SimpleNamespace(shape=(1, 3, h, w)), types.SimpleNamespace(shape=(h, w, 3)), "x")
        return np.asarray(out).reshape(-1, 5)


def occlusion_risk(boxes):
    """Largest fraction of each box covered by another box (self excluded)."""
    from .geometry import overlap
    if len(boxes) < 2:
        return np.zeros(len(boxes))
    ioa = overlap(boxes, boxes, mode="ioa")
    np.fill_diagonal(ioa, 0)
    return ioa.max(axis=1)


def assoc_softmax(x, axis):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def _key(feat):
    """Stable lookup key for an embedding row (robust to re-normalisation round-off)."""
    f = np.asarray(feat, float)
    return np.round(f / max(np.linalg.norm(f), 1e-12), 5).tobytes()


def orm_alpha(alpha_old, risk, kappa):
    """Occlusion-Risk Memory: shrink the *new-sample* weight (1-alpha_old) by occlusion risk, never to 0."""
    return 1 - (1 - alpha_old) * max(0.1, 1 - kappa * risk)


def gaw_boost(iou, kappa, margin):
    """Geometry-Ambiguity Weighting: 0..kappa boost where the best and second-best IoU are close."""
    def m(g, axis):
        if g.shape[axis] < 2:
            return np.ones(g.shape[1 - axis])
        s = np.sort(g, axis=axis)
        return s.take(-1, axis=axis) - s.take(-2, axis=axis)
    mm = np.minimum(m(iou, 1)[:, None], m(iou, 0)[None, :])
    return kappa * (1 - np.clip(mm / margin, 0, 1))


class DeepOCSortRFAdapter(DeepOCSortAdapter):
    """Deep OC-SORT + ORM (memory) + GAW (association weight). Upstream file untouched; hooks patched."""

    def __init__(self, params, fps, mot20):
        p = dict(params)
        self.kappa_risk = p.pop("orm_kappa", 0.9)
        self.kappa_gaw = p.pop("gaw_kappa", 0.5)
        self.gaw_margin = p.pop("gaw_margin", 0.3)
        self.bisoftmax_beta = p.pop("bisoftmax_beta", 0.0)
        self.temperature = p.pop("temperature", 0.1)
        self.recovery_cos = p.pop("recovery_cos", 0.0)
        self.recovery_dist = p.pop("recovery_dist", 1.5)
        self._emb_override = None
        super().__init__(p, fps, mot20)
        from trackers.integrated_ocsort_embedding import association as assoc, ocsort as docs
        adapter = self
        original_aw = assoc.compute_aw_new_metric
        original_associate = assoc.associate

        def associate(detections, trackers, det_embs, trk_embs, iou_threshold, *args, **kwargs):
            adapter._iou = assoc.iou_batch(detections, trackers) if len(trackers) and len(detections) else None
            beta = adapter.bisoftmax_beta
            if beta and adapter._iou is not None and len(det_embs) and len(trk_embs):
                # MASA bi-softmax: mutual-exclusivity normalisation of the cosine matrix
                cos = det_embs @ trk_embs.T
                adapter._emb_override = (1 - beta) * cos + beta * (
                    (assoc_softmax(cos / adapter.temperature, 0) + assoc_softmax(cos / adapter.temperature, 1)) / 2)
            else:
                adapter._emb_override = None
            matches, um_d, um_t = original_associate(detections, trackers, det_embs, trk_embs, iou_threshold,
                                                     *args, **kwargs)
            if adapter.recovery_cos and len(um_d) and len(um_t) and len(det_embs) and len(trk_embs):
                # appearance recovery: unmatched pairs with strong appearance and plausible centre distance
                d_idx, t_idx = np.asarray(um_d, int), np.asarray(um_t, int)
                cos = det_embs[d_idx] @ trk_embs[t_idx].T
                dc = (detections[d_idx, None, :2] + detections[d_idx, None, 2:4]) / 2
                tc = (trackers[None, t_idx, :2] + trackers[None, t_idx, 2:4]) / 2
                h = np.maximum(detections[d_idx, 3] - detections[d_idx, 1], 1)[:, None]
                ok = (cos >= adapter.recovery_cos) & (np.linalg.norm(dc - tc, axis=2) <= adapter.recovery_dist * h)
                if ok.any():
                    rows, cols = assoc.linear_assignment(-np.where(ok, cos, -10.0)).T if ok.any() else ([], [])
                    extra = [(d_idx[r], t_idx[c]) for r, c in zip(rows, cols) if ok[r, c]]
                    if extra:
                        matches = np.concatenate([matches.reshape(-1, 2), np.array(extra, int)])
                        used_d, used_t = {e[0] for e in extra}, {e[1] for e in extra}
                        um_d = np.array([d for d in um_d if d not in used_d])
                        um_t = np.array([t for t in um_t if t not in used_t])
            return matches, um_d, um_t

        def compute_aw(emb_cost, w, max_diff=0.5):
            if adapter._emb_override is not None and adapter._emb_override.shape == emb_cost.shape:
                emb_cost[...] = adapter._emb_override  # caller multiplies this array in place by the weights
            base = original_aw(emb_cost, w, max_diff)
            if adapter.kappa_gaw and adapter._iou is not None and adapter._iou.shape == emb_cost.shape:
                base = base + w * gaw_boost(adapter._iou, adapter.kappa_gaw, adapter.gaw_margin)
            return base

        def update_emb(trk, emb, alpha=0.9):
            r = adapter.risk.get(_key(emb), 0.0)
            a = orm_alpha(alpha, r, adapter.kappa_risk) if adapter.kappa_risk else alpha
            trk.emb = a * trk.emb + (1 - a) * emb
            trk.emb /= np.linalg.norm(trk.emb)

        assoc.compute_aw_new_metric = compute_aw
        docs.associate = associate
        docs.KalmanBoxTracker.update_emb = update_emb
        self._iou, self.risk = None, {}

    def update(self, boxes, scores, embs, warp, hw):
        keep = scores > self.det_thresh
        feats = _norm(embs[keep]) if keep.any() else np.empty((0, 256))
        risk = occlusion_risk(boxes[keep]) if keep.any() else np.zeros(0)
        self.risk = {_key(f): float(r) for f, r in zip(feats, risk)}
        return super().update(boxes, scores, embs, warp, hw)


class BoTSortRFAdapter(BoTSortAdapter):
    """BoT-SORT(ReID) + ORM: feature EMA new-sample weight shrinks with occlusion risk."""

    def __init__(self, params, fps, mot20):
        p = dict(params)
        self.kappa_risk = p.pop("orm_kappa", 0.9)
        super().__init__(p, fps, mot20, reid=True)
        from tracker import bot_sort
        adapter = self

        def update_features(trk, feat):
            feat = feat / np.linalg.norm(feat)
            trk.curr_feat = feat
            r = adapter.risk.get(_key(feat), 0.0)
            if trk.smooth_feat is None:
                trk.smooth_feat = feat
            else:
                a = orm_alpha(getattr(trk, "alpha", 0.9), r, adapter.kappa_risk)
                trk.smooth_feat = a * trk.smooth_feat + (1 - a) * feat
            trk.features.append(feat)
            trk.smooth_feat /= np.linalg.norm(trk.smooth_feat)

        bot_sort.STrack.update_features = update_features
        self.risk = {}

    def update(self, boxes, scores, embs, warp, hw):
        a = self.args
        low = scores > a.track_low_thresh
        high = scores[low] > a.track_high_thresh
        b = boxes[low][high]
        feats = _norm(embs[low][high]) if len(embs) else np.empty((0, 256))
        self.risk = {_key(f): float(r) for f, r in zip(feats, occlusion_risk(b))}
        return super().update(boxes, scores, embs, warp, hw)


class RiskFusionAdapter:
    def __init__(self, params, fps, mot20):
        from .det_tracker import RFConfig, RiskFusionTracker
        p = dict(params)
        if mot20:
            p.setdefault("fuse_score", False)
        self.t = RiskFusionTracker(RFConfig(**p), fps=fps)

    def update(self, boxes, scores, embs, warp, hw):
        return self.t.update(boxes, scores, embs, warp)


class MasaAdapter:
    """MASA association logic (ours re-implementation, greedy or Hungarian) on detector boxes."""

    def __init__(self, params, fps, mot20):
        from dataclasses import replace
        from .config import TrackerConfig
        from .tracker import ReliabilityTracker
        p = {"variant": "masa", "init_score_threshold": 0.8, "object_score_threshold": 0.5,
             "match_threshold": 0.5, "max_age": 10, "max_distance": 100.0, **params}
        self.t = ReliabilityTracker(replace(TrackerConfig(), fps=fps, **p))
        self.frame = 0

    def update(self, boxes, scores, embs, warp, hw):
        self.frame += 1
        ids = self.t.update(self.frame, boxes, embs, scores, np.zeros(len(boxes), int), "none")
        keep = ids > 0
        return np.c_[boxes[keep], ids[keep]].reshape(-1, 5)


class AuthorOAAdapter:
    """Unmodified MASA-OA tracker source (authors) on detector boxes."""

    def __init__(self, params, fps, mot20):
        import torch
        from mmengine.structures import InstanceData
        from .upstream_tracker import load_author_tracker
        p = {"init_score_thr": 0.8, "obj_score_thr": 0.5, "match_score_thr": 0.5, "memo_tracklet_frames": 10,
             "memo_momentum": 0.8, "max_distance": 100.0, **params}
        source_root = p.pop("source_root", None)  # patched copy of the author file, for mechanism ablations
        self.model = load_author_tracker(ROOT / source_root if source_root else TP / "MASA-OA")(
            fps=fps, with_cats=False, **p)
        self.torch, self.InstanceData, self.frame = torch, InstanceData, 0

    def update(self, boxes, scores, embs, warp, hw):
        torch = self.torch
        b = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        e = torch.as_tensor(embs, dtype=torch.float32).reshape(-1, 256)
        s = torch.as_tensor(scores, dtype=torch.float32)
        pred = self.InstanceData(bboxes=b, scores=s, labels=torch.zeros(len(b), dtype=torch.long))
        sample = types.SimpleNamespace(pred_instances=pred, metainfo={"frame_id": self.frame, "scale_factor": (1, 1)})
        self.frame += 1
        head = types.SimpleNamespace(predict=lambda feats, rois: e)
        out = self.model.track(types.SimpleNamespace(track_head=head), None, [], sample)
        return np.c_[out.bboxes.numpy(), out.instances_id.numpy() + 1].reshape(-1, 5)


def make(method, params, fps, mot20):
    if method == "bytetrack":
        return ByteTrackAdapter(params, fps, mot20)
    if method == "ocsort":
        return OCSortAdapter(params, fps, mot20)
    if method == "botsort":
        return BoTSortAdapter(params, fps, mot20, reid=False)
    if method == "botsort-reid":
        return BoTSortAdapter(params, fps, mot20, reid=True)
    if method == "deepocsort":
        return DeepOCSortAdapter(params, fps, mot20)
    if method == "deepocsort-rf":
        return DeepOCSortRFAdapter(params, fps, mot20)
    if method == "botsort-reid-rf":
        return BoTSortRFAdapter(params, fps, mot20)
    if method == "masa-rf":
        return RiskFusionAdapter(params, fps, mot20)
    if method == "masa":
        return MasaAdapter(params, fps, mot20)
    if method == "masa-oa-author":
        return AuthorOAAdapter(params, fps, mot20)
    raise ValueError(f"Unknown method {method}")
