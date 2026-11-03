import numpy as np
import pytest
from era_masa import ReliabilityTracker, TrackerConfig


def test_camera_shift_changes_motion_coordinates_but_preserves_output_id_order():
    tracker = ReliabilityTracker(TrackerConfig(variant="robust-fusion", camera_motion=True,
        appearance_weight=0.1, max_distance=10, center_weight=0.2))
    boxes = np.array([[0., 0., 20., 60.], [200., 0., 220., 60.]])
    features = np.eye(256)[:2] * 8
    ids = tracker.update(1, boxes, features, camera_translation=np.zeros(2))
    shifted = (boxes + [80, 10, 80, 10])[::-1]
    result = tracker.update(2, shifted, features[::-1], camera_translation=np.array([80, 10]))
    np.testing.assert_array_equal(result, ids[::-1])
    for track in tracker.tracks.values():
        np.testing.assert_allclose(track.velocity, np.zeros(4))


def test_camera_cache_is_required_when_enabled():
    tracker = ReliabilityTracker(TrackerConfig(variant="robust-fusion", camera_motion=True))
    with pytest.raises(ValueError, match="without measured"):
        tracker.update(1, np.empty((0, 4)), np.empty((0, 256)))


def test_smoothed_center_velocity_ignores_box_size_noise():
    tracker = ReliabilityTracker(TrackerConfig(variant="robust-fusion", velocity_alpha=0.5,
                                              appearance_weight=0.9, max_distance=-1))
    feature = np.eye(256)[:1] * 8
    tid = tracker.update(1, np.array([[0., 0., 20., 60.]]), feature)[0]
    tracker.update(2, np.array([[-3., -5., 27., 65.]]), feature)
    np.testing.assert_allclose(tracker.tracks[tid].velocity, [2, 0, 2, 0])
    tracker.update(3, np.array([[3., -3., 33., 67.]]), feature)
    np.testing.assert_allclose(tracker.tracks[tid].velocity, [4, 1, 4, 1])
