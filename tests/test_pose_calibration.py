"""test_pose_calibration.py
Tests for scanning, capturing, saving, and matching custom hand poses.
"""
import os
import tempfile
from pose_calibration import PoseManager
from tests.test_gestures import dummy_landmarks

def test_pose_manager_capture_and_match():
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        pm = PoseManager(filepath=tmp_path)
        lm_custom = dummy_landmarks({4: (100, 100), 8: (105, 105)})

        # Capture pose
        success = pm.capture_pose("test_gesture", lm_custom)
        assert success is True
        assert "test_gesture" in pm.poses

        # Test exact match
        matched, dist = pm.match_pose(lm_custom, "test_gesture", threshold=0.1)
        assert matched is True
        assert dist < 0.05

        # Test save and reload
        assert pm.save_poses() is True

        pm2 = PoseManager(filepath=tmp_path)
        assert "test_gesture" in pm2.poses
        matched2, _ = pm2.match_pose(lm_custom, "test_gesture")
        assert matched2 is True
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

