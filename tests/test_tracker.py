from dataclasses import replace
import numpy as np
import pytest
from era_masa import ReliabilityTracker, TrackerConfig, VARIANTS
from era_masa.assignment import assign
from era_masa.geometry import crowding
from era_masa.kalman import BoxKalman
from era_masa.reliability import estimate


def embeddings(n=2):
    return np.eye(256)[:n] * 8


@pytest.mark.parametrize("variant", VARIANTS)
def test_clean_tracks_keep_identity_and_detection_order(variant):
    tracker = ReliabilityTracker(TrackerConfig(variant=variant))
    boxes = np.array([[0, 0, 20, 60], [200, 0, 220, 60]], dtype=float)
    first = tracker.update(1, boxes, embeddings())
    for frame in range(2, 8):
        # Reverse detection order and confidence, ensuring IDs are returned in
        # input order and greedy/Hungarian operate on the same underlying data.
        current = tracker.update(frame, boxes[::-1] + [frame, 0, frame, 0],
                                 embeddings()[::-1], [0.5, 1.0])
        assert np.array_equal(current, first[::-1])
        assert len(np.unique(current)) == 2


def test_invalid_edge_is_gated_before_hungarian():
    cost = np.array([[0.0, 0.4], [0.1, 0.2]])
    valid = np.array([[False, True], [True, False]])
    assert assign(cost, valid, 0.5) == [(0, 1), (1, 0)]


def test_dummy_assignments_allow_abstaining():
    assert assign(np.ones((3, 2)), np.ones((3, 2), dtype=bool), 0.5) == []


def test_oa_embedding_is_frozen_once_per_frame():
    tracker = ReliabilityTracker(TrackerConfig(variant="masa-oa"))
    boxes = np.array([[0, 0, 20, 60], [5, 0, 25, 60]])
    ids = tracker.update(1, boxes, embeddings())
    old = {tid: t.embedding.copy() for tid, t in tracker.tracks.items()}
    assert np.array_equal(tracker.update(2, boxes, embeddings()[::-1]), ids)
    assert tracker.statistics["frozen_updates"] == 2
    for tid in ids:
        np.testing.assert_array_equal(tracker.tracks[tid].embedding, old[tid])
    assert all(row["alpha"] == 0 for row in tracker.diagnostics)


def test_reliability_memory_gate_freezes_under_crowding():
    cfg = TrackerConfig(weights={"temporal": 0, "rival": 0, "geometry": 1, "motion": 0})
    tracker = ReliabilityTracker(cfg)
    boxes = np.array([[0, 0, 20, 60], [3, 0, 23, 60]])
    ids = tracker.update(1, boxes, embeddings())
    old = tracker.tracks[int(ids[0])].embedding.copy()
    tracker.update(2, boxes, embeddings()[::-1])
    # This test verifies memory protection even if ambiguous matching swaps IDs.
    # It must not assume an uncalibrated ECR always resolves overlapping boxes.
    assert tracker.statistics["frozen_updates"] == 2
    np.testing.assert_array_equal(tracker.tracks[int(ids[0])].embedding, old)


def test_neighbor_signal_requires_local_competing_track():
    cfg = TrackerConfig(variant="temporal-rival")
    boxes = np.array([[0, 0, 20, 60]])
    predicted = np.array([[0, 0, 20, 60], [10000, 0, 10020, 60]])
    memory = embeddings()
    poisoned = memory[1:2]
    result = estimate(boxes, poisoned, predicted, memory, np.ones(2, dtype=bool), np.zeros((1, 2)), cfg)
    assert result.signals["rival"][0, 0] == 0
    predicted[1] = [5, 0, 25, 60]
    result = estimate(boxes, poisoned, predicted, memory, np.ones(2, dtype=bool), np.zeros((1, 2)), cfg)
    assert result.signals["rival"][0, 0] == 1


def test_zero_embedding_never_updates_memory():
    tracker = ReliabilityTracker()
    boxes = np.array([[0, 0, 20, 60]])
    ids = tracker.update(1, boxes, embeddings(1))
    before = tracker.tracks[int(ids[0])].embedding.copy()
    assert np.array_equal(tracker.update(2, boxes, np.zeros((1, 256))), ids)
    np.testing.assert_array_equal(tracker.tracks[int(ids[0])].embedding, before)
    assert tracker.diagnostics[0]["reliability"] == 0


def test_empty_frames_age_and_delete_tracks():
    tracker = ReliabilityTracker(TrackerConfig(max_age=2))
    boxes = np.array([[0, 0, 20, 60]])
    first = tracker.update(1, boxes, embeddings(1))
    assert len(tracker.update(2, np.empty((0, 4)), np.empty((0, 256)))) == 0
    second = tracker.update(4, boxes, embeddings(1))
    assert first[0] != second[0]
    assert len(tracker.tracks) == 1


def test_frame_gap_advances_same_kalman_prediction():
    a = ReliabilityTracker()
    b = ReliabilityTracker()
    boxes = np.array([[0, 0, 20, 60]])
    for tracker in (a, b):
        tracker.update(1, boxes, embeddings(1))
        tracker.update(2, boxes + [2, 0, 2, 0], embeddings(1))
    a.update(3, np.empty((0, 4)), np.empty((0, 256)))
    a.update(4, np.empty((0, 4)), np.empty((0, 256)))
    b.update(4, np.empty((0, 4)), np.empty((0, 256)))
    np.testing.assert_allclose(a.tracks[1].kf.mean, b.tracks[1].kf.mean)
    np.testing.assert_allclose(a.tracks[1].kf.covariance, b.tracks[1].kf.covariance)


def test_normalized_reliability_and_singleton_self_overlap():
    boxes = np.array([[0, 0, 20, 60]])
    assert crowding(boxes)[0][0] == 0
    cfg = TrackerConfig()
    result = estimate(boxes, embeddings(1), boxes, embeddings(1), np.ones(1, dtype=bool), np.zeros((1, 1)), cfg)
    assert result.reliability[0, 0] == pytest.approx(1)


@pytest.mark.parametrize("variant", VARIANTS)
def test_category_gate_is_respected(variant):
    tracker = ReliabilityTracker(TrackerConfig(variant=variant, with_categories=True))
    boxes = np.array([[0, 0, 20, 60]])
    first = tracker.update(1, boxes, embeddings(1), labels=[0])
    second = tracker.update(2, boxes, embeddings(1), labels=[1])
    assert first[0] != second[0]


@pytest.mark.parametrize("failure", ["nan", "dimension", "negative_box", "frame_order"])
def test_invalid_input_is_rejected(failure):
    tracker = ReliabilityTracker()
    boxes = np.array([[0, 0, 20, 60]], dtype=float)
    tracker.update(1, boxes, embeddings(1))
    frame, features = 2, embeddings(1)
    if failure == "nan":
        features[0, 0] = np.nan
    if failure == "dimension":
        features = np.zeros((1, 128))
    if failure == "negative_box":
        boxes[0, 2] = -1
    if failure == "frame_order":
        frame = 1
    with pytest.raises(ValueError):
        tracker.update(frame, boxes, features)


def test_kalman_covariance_stays_positive():
    kf = BoxKalman([10, 30, 1/3, 60])
    for frame in range(200):
        kf.predict()
        kf.update([10+frame, 30, 1/3, 60])
        assert np.linalg.eigvalsh(kf.covariance).min() > 0


def test_no_diagnostics_keeps_accurate_statistics():
    tracker = ReliabilityTracker()
    boxes = np.array([[0, 0, 20, 60]])
    tracker.update(1, boxes, embeddings(1), diagnostics="none")
    assert tracker.statistics["new_tracks"] == 1
    tracker.update(2, boxes, embeddings(1), diagnostics="none")
    assert tracker.statistics["matches"] == 1
    assert tracker.diagnostics == []
