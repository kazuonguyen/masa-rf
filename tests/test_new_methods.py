import numpy as np
import pytest
from era_masa import ReliabilityTracker, TrackerConfig


@pytest.mark.parametrize("variant", ("hybrid", "cascade", "gallery", "observation"))
def test_new_families_follow_objects_through_reordered_detections_and_gap(variant):
    tracker = ReliabilityTracker(TrackerConfig(variant=variant, max_distance=-1))
    features = np.eye(256)[:2] * 8
    boxes = np.array([[0, 0, 20, 60], [200, 0, 220, 60]], dtype=float)
    ids = tracker.update(1, boxes, features)
    tracker.update(2, boxes + [2, 0, 2, 0], features)
    tracker.update(3, np.empty((0, 4)), np.empty((0, 256)))
    recovered = tracker.update(4, (boxes + [6, 0, 6, 0])[::-1], features[::-1])
    np.testing.assert_array_equal(recovered, ids[::-1])


def test_gallery_stays_bounded_and_rejects_invalid_embedding():
    tracker = ReliabilityTracker(TrackerConfig(variant="gallery", gallery_size=3))
    box = np.array([[0., 0., 20., 60.]])
    feature = np.eye(256)[:1] * 8
    tid = tracker.update(1, box, feature)[0]
    for frame in range(2, 10):
        tracker.update(frame, box, feature)
    assert len(tracker.tracks[tid].gallery) == 3
    before = [v.copy() for v in tracker.tracks[tid].gallery]
    tracker.update(10, box, np.zeros_like(feature))
    for left, right in zip(before, tracker.tracks[tid].gallery):
        np.testing.assert_array_equal(left, right)
