"""T3 Hardware Tests: xArm 6 (192.168.1.60) read-only telemetry, Eye-in-Hand calibration, and Gemini 305 capture."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest
import yaml

from core.transforms import load_hand_eye_calibration, verify_calibration_orthonormality
from core.types import GRIP_DOWN_QUAT_WXYZ, Pose
from execution.xarm_backend import XArmBackend, XArmSafetyError
from perception.orbbec_gemini import OrbbecPerception

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CALIB_PATH = PROJECT_ROOT / "configs" / "camera_calibration.yaml"


def test_xarm6_readonly_telemetry_and_safety_gate() -> None:
    """Connect to real xArm 6 (192.168.1.60) in read-only mode, verify latency < 20ms and motion lock."""
    backend = XArmBackend(ip="192.168.1.60", read_only=True)
    try:
        code_sn, sn = backend.get_robot_sn()
        assert code_sn == 0
        assert sn.startswith("XI"), f"Unexpected xArm serial number: {sn}"

        t0 = time.perf_counter()
        state = backend.get_robot_state()
        latency_ms = (time.perf_counter() - t0) * 1000.0

        assert latency_ms < 20.0, f"xArm telemetry latency {latency_ms:.2f} ms >= 20 ms"
        assert len(state.joint_positions_rad) == 6
        assert np.all(np.isfinite(state.joint_positions_rad))
        assert np.all(np.isfinite(state.tcp_pose.pos_np))

        code_pos, pos_mm_deg = backend.get_position(is_radian=False)
        assert code_pos == 0 and len(pos_mm_deg) == 6

        # Verify fail-closed safety gate blocks any motion command in read_only=True mode
        with pytest.raises(XArmSafetyError, match="read_only=True"):
            backend.reset()
        with pytest.raises(XArmSafetyError, match="read_only=True"):
            backend.set_position(x=300.0, y=0.0, z=200.0, roll=180.0, pitch=0.0, yaw=0.0)
        with pytest.raises(XArmSafetyError, match="read_only=True"):
            backend.move_tcp(Pose.from_xyz(0.30, 0.0, 0.20, GRIP_DOWN_QUAT_WXYZ))
        with pytest.raises(XArmSafetyError, match="read_only=True"):
            backend.open_gripper()
    finally:
        backend.disconnect()


def test_hand_eye_calibration_metrics() -> None:
    """Verify Eye-in-Hand calibration YAML accuracy (< 5.0 mm mean/max error) and SO(3) validity."""
    calib = load_hand_eye_calibration(CALIB_PATH)
    assert verify_calibration_orthonormality(calib, tol=1e-5)

    raw = yaml.safe_load(CALIB_PATH.read_text(encoding="utf-8"))
    metrics = raw["metrics"]
    assert float(metrics["mean_mm"]) < 5.0
    assert float(metrics["max_mm"]) < 5.0
    assert float(metrics["reprojection_rms_px"]) < 1.5


def test_orbbec_gemini305_rgbd_capture() -> None:
    """Capture a live aligned RGB-D frame from the connected Orbbec Gemini 305 camera."""
    backend = XArmBackend(ip="192.168.1.60", read_only=True)
    try:
        perception = OrbbecPerception(backend, CALIB_PATH, width=640, height=480, fps=30)
        bgr, depth_m = perception.capture_aligned_rgbd(timeout_ms=2500)

        assert bgr.shape == (480, 640, 3)
        assert bgr.dtype == np.uint8
        assert depth_m.shape == (480, 640)
        assert depth_m.dtype == np.float32
        assert np.any(bgr > 0), "Captured BGR image is completely black"
        assert np.any(depth_m > 0.0), "Captured aligned depth map has no valid depth pixels"
    finally:
        backend.disconnect()
