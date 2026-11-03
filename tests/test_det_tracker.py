import numpy as np
from era_masa.det_tracker import RFConfig, RiskFusionTracker, _apply_affine
from era_masa.det_run import interpolate, postfilter
from era_masa.kalman import BoxKalman
from era_masa.shims.cython_bbox import bbox_overlaps
from tools.prepare_gt import half_offset


def _emb(i, dim=256):
    e = np.zeros(dim)
    e[i] = 1.0
    return e


def test_crossing_objects_keep_identity_with_distinct_appearance():
    t = RiskFusionTracker(RFConfig(), fps=30)
    ids = {}
    for f in range(40):
        a = np.array([10 + 5 * f, 100, 50 + 5 * f, 200.0])
        b = np.array([210 - 5 * f, 105, 250 - 5 * f, 205.0])
        out = t.update(np.stack([a, b]), np.array([0.9, 0.9]), np.stack([_emb(0), _emb(1)]))
        for row in out:
            key = 0 if np.allclose(row[:4], a) else 1
            ids.setdefault(key, set()).add(int(row[4]))
    assert ids[0] and ids[1] and len(ids[0]) == 1 and len(ids[1]) == 1 and ids[0] != ids[1]


def test_low_score_detection_never_starts_track_or_writes_memory():
    t = RiskFusionTracker(RFConfig(), fps=30)
    t.update(np.array([[0, 0, 10, 20.0]]), np.array([0.9]), _emb(0)[None])
    before = t.tracks[0].emb.copy()
    out = t.update(np.array([[0, 0, 10, 20.0], [100, 0, 110, 20]]), np.array([0.3, 0.3]),
                   np.stack([_emb(5), _emb(6)]))
    assert len(t.tracks) == 1 and np.allclose(t.tracks[0].emb, before) and len(out) == 1


def test_unconfirmed_track_dies_after_one_miss_and_lost_track_expires():
    t = RiskFusionTracker(RFConfig(track_buffer=3), fps=30)
    t.update(np.array([[0, 0, 10, 20.0]]), np.array([0.9]), _emb(0)[None])
    t.update(np.array([[200, 0, 210, 20.0]]), np.array([0.9]), _emb(1)[None])  # new unconfirmed + lost
    t.update(np.empty((0, 4)), np.empty(0), np.empty((0, 256)))
    assert [tr.tid for tr in t.tracks] == [1]
    for _ in range(4):
        t.update(np.empty((0, 4)), np.empty(0), np.empty((0, 256)))
    assert t.tracks == []


def test_camera_translation_moves_kalman_state():
    kf = BoxKalman(np.array([50, 60, 0.5, 100.0]))
    _apply_affine(kf, np.array([[1, 0, 7], [0, 1, -3.0]]))
    assert np.allclose(kf.mean[:4], [57, 57, 0.5, 100])


def test_cython_bbox_shim_uses_plus_one_convention():
    iou = bbox_overlaps(np.array([[0, 0, 9, 9.]]), np.array([[0, 0, 9, 9], [5, 5, 14, 14.]]))
    assert np.allclose(iou, [[1, 25 / 175]])


def test_interpolation_and_output_filter():
    rows = np.array([[1, 1, 0, 0, 10, 20], [4, 1, 3, 0, 13, 20.]])
    filled = interpolate(rows)
    assert len(filled) == 4 and np.allclose(filled[1, 2:], [1, 0, 11, 20])
    wide = np.array([[1, 2, 0, 0, 40, 10.]])
    assert len(postfilter(wide, 10, aspect=True)) == 0 and len(postfilter(wide, 10, aspect=False)) == 1


def test_bytetrack_half_split_is_disjoint():
    for n in (600, 601, 1050):
        train_end, val_start = half_offset(n), half_offset(n) + 1
        assert train_end < val_start and val_start == n // 2 + 2


def test_out_of_frame_boxes_survive_cache_and_output_filter(tmp_path):
    """Regression for the v1 clipping bug: boxes extending beyond the image must reach trackers unchanged."""
    from era_masa.detcache import load_det_cache, save_det_cache
    boxes = np.array([[548.4, -41.5, 592.3, 92.6], [311.2, 918.7, 419.2, 1198.1]])
    save_det_cache(tmp_path / "s.npz", metadata={"sequence": "s", "length": 1, "fps": 30.0, "width": 1920, "height": 1080},
                   frames=np.array([1, 1]), boxes=boxes, scores=np.array([0.9, 0.9]),
                   embeddings=np.ones((2, 256)), gmc=np.zeros((1, 2, 3)))
    seq = load_det_cache(tmp_path / "s.npz")
    assert np.allclose(seq.boxes, boxes, atol=1e-3)
    rows = np.c_[[1, 1], [1, 2], boxes]
    assert np.allclose(postfilter(rows, 10, aspect=True)[:, 2:], boxes)


def test_box_convention_follows_benchmark_gt():
    """MOT17 GT extends beyond the frame (keep detector boxes); MOT20/DanceTrack GT is in-frame (clip)."""
    from era_masa.det_run import CLIP_BENCHMARKS, clip_boxes
    assert "MOT17" not in CLIP_BENCHMARKS and {"MOT20", "DanceTrack"} <= set(CLIP_BENCHMARKS)
    b = clip_boxes(np.array([[-5, -40, 30, 1200.0]]), 1920, 1080)
    assert np.allclose(b, [[0, 0, 30, 1079]])


def test_hard_freeze_keeps_memory_only_when_occlusion_risk_is_high():
    box_a, box_b = np.array([0, 0, 100, 200.0]), np.array([10, 0, 110, 200.0])
    far = np.array([500, 0, 600, 200.0])
    frozen = RiskFusionTracker(RFConfig(risk_memory=0.0, ema_alpha=0.5, freeze_risk=0.3), fps=30)
    frozen.update(np.stack([box_a, far]), np.array([0.9, 0.9]), np.stack([_emb(0), _emb(2)]))
    before = {tuple(t.box): t.emb.copy() for t in frozen.tracks}
    frozen.update(np.stack([box_b, far]), np.array([0.9, 0.9]), np.stack([_emb(0) + _emb(3), _emb(2) + _emb(4)]))
    # no other box overlaps either detection, so both memories still update
    assert all(not np.allclose(t.emb, before[k]) for t, k in zip(frozen.tracks, before))
    # a covered detection keeps the track memory unchanged
    occluder = np.array([0, 0, 100, 200.0])
    memory = frozen.tracks[0].emb.copy()
    frozen.update(np.stack([box_b, occluder]), np.array([0.9, 0.9]), np.stack([_emb(7), _emb(8)]))
    assert np.allclose(frozen.tracks[0].emb, memory)
