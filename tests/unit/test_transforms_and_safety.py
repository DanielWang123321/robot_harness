"""T1 Unit Tests: SE(3) coordinate transforms, Eye-in-Hand chain, RGB-D localization, and WorkspaceValidator."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from core.safety import WorkspaceValidator, WorkspaceViolationError
from core.transforms import (
    camera_point_to_base,
    load_hand_eye_calibration,
    matrix_to_pose,
    matrix_to_quat_wxyz,
    matrix_to_rpy,
    pixel_depth_to_camera_point,
    pose_to_matrix,
    quat_wxyz_to_matrix,
    rpy_to_matrix,
    verify_calibration_orthonormality,
)
from core.types import GRIP_DOWN_QUAT_WXYZ, Pose, WorkspaceLimits
from perception.orbbec_gemini import detect_color_objects_rgbd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CALIB_PATH = PROJECT_ROOT / "configs" / "camera_calibration.yaml"


def test_pose_and_rotation_roundtrip() -> None:
    """Verify rotation matrix, quaternion (wxyz), RPY, and Pose conversions have error < 1e-6."""
    rng = np.random.default_rng(42)
    for _ in range(25):
        angles = rng.uniform(-np.pi / 2, np.pi / 2, size=3)
        R = rpy_to_matrix(float(angles[0]), float(angles[1]), float(angles[2]))
        r2, p2, y2 = matrix_to_rpy(R)
        R_reconstructed = rpy_to_matrix(r2, p2, y2)
        assert np.max(np.abs(R - R_reconstructed)) < 1e-6

        q = matrix_to_quat_wxyz(R)
        R_from_q = quat_wxyz_to_matrix(q)
        assert np.max(np.abs(R - R_from_q)) < 1e-6

        xyz = rng.uniform(-0.5, 0.5, size=3)
        pose = Pose.from_xyz(float(xyz[0]), float(xyz[1]), float(xyz[2]), q)
        T = pose_to_matrix(pose)
        pose_back = matrix_to_pose(T)
        assert np.linalg.norm(pose.pos_np - pose_back.pos_np) < 1e-6
        assert np.max(np.abs(quat_wxyz_to_matrix(pose.quat_np) - quat_wxyz_to_matrix(pose_back.quat_np))) < 1e-6


def test_hand_eye_chain_and_backprojection() -> None:
    """Verify pixel + depth -> camera frame -> base frame transform accuracy (< 1e-6)."""
    calib = load_hand_eye_calibration(CALIB_PATH)
    assert verify_calibration_orthonormality(calib, tol=1e-5)

    # Known 3D point in camera frame
    p_cam_true = np.array([0.04, -0.03, 0.35], dtype=float)
    fx, fy = calib.K[0, 0], calib.K[1, 1]
    cx, cy = calib.K[0, 2], calib.K[1, 2]
    u = fx * (p_cam_true[0] / p_cam_true[2]) + cx
    v = fy * (p_cam_true[1] / p_cam_true[2]) + cy
    depth_m = p_cam_true[2]

    p_cam_est = pixel_depth_to_camera_point(u, v, depth_m, calib.K)
    assert np.linalg.norm(p_cam_est - p_cam_true) < 1e-6

    flange_pose = Pose.from_xyz(0.30, 0.05, 0.32, GRIP_DOWN_QUAT_WXYZ)
    T_base_flange = pose_to_matrix(flange_pose)
    p_base = camera_point_to_base(p_cam_est, T_base_flange, calib.T_flange_cam)

    # Reverse transform back to camera frame
    T_base_cam = T_base_flange @ calib.T_flange_cam
    p_cam_recovered = (np.linalg.inv(T_base_cam) @ np.append(p_base, 1.0))[:3]
    assert np.linalg.norm(p_cam_recovered - p_cam_true) < 1e-6


def test_synthetic_rgbd_object_detection() -> None:
    """Verify RGB-D color segmentation + 3D localization in base frame."""
    calib = load_hand_eye_calibration(CALIB_PATH)
    h, w = calib.height, calib.width
    bgr = np.zeros((h, w, 3), dtype=np.uint8)
    depth_m = np.full((h, w), 0.40, dtype=np.float32)

    # Draw a red square centered at (220, 200) at depth 0.32 m
    cv2.rectangle(bgr, (200, 180), (240, 220), (0, 0, 230), -1)
    depth_m[180:220, 200:240] = 0.32

    # Draw a green circle centered at (420, 280) at depth 0.34 m
    cv2.circle(bgr, (420, 280), 20, (0, 220, 0), -1)
    depth_m[260:300, 400:440] = 0.34

    T_base_flange = pose_to_matrix(Pose.from_xyz(0.30, 0.0, 0.35, GRIP_DOWN_QUAT_WXYZ))
    detected = detect_color_objects_rgbd(bgr, depth_m, calib.K, T_base_flange, calib.T_flange_cam)

    assert "red_cube" in detected
    assert "green_target" in detected

    expected_red_cam = pixel_depth_to_camera_point(220.0, 200.0, 0.32, calib.K)
    expected_red_base = camera_point_to_base(expected_red_cam, T_base_flange, calib.T_flange_cam)
    assert np.linalg.norm(detected["red_cube"].pose.pos_np - expected_red_base) < 0.002


def test_workspace_validator_intercepts_all_violations() -> None:
    """Verify 100% interception of out-of-bounds XYZ, table collision Z < z_min, tilt, and speed."""
    limits = WorkspaceLimits(
        x_min_m=0.15,
        x_max_m=0.65,
        y_min_m=-0.45,
        y_max_m=0.45,
        z_min_m=0.008,
        z_max_m=0.55,
        max_tilt_deg=10.0,
        max_cart_vel_m_s=0.25,
    )
    validator = WorkspaceValidator(limits)

    # Valid pose passes
    validator.validate_pose(Pose.from_xyz(0.30, 0.10, 0.05, GRIP_DOWN_QUAT_WXYZ))
    assert validator.validate_speed(0.15) == pytest.approx(0.15)

    # Table collision Z < z_min_m
    with pytest.raises(WorkspaceViolationError, match="table collision guard"):
        validator.validate_pose(Pose.from_xyz(0.30, 0.0, 0.002, GRIP_DOWN_QUAT_WXYZ))

    # Out of X range
    with pytest.raises(WorkspaceViolationError, match="X="):
        validator.validate_pose(Pose.from_xyz(0.80, 0.0, 0.15, GRIP_DOWN_QUAT_WXYZ))

    # Out of Y range
    with pytest.raises(WorkspaceViolationError, match="Y="):
        validator.validate_pose(Pose.from_xyz(0.30, -0.60, 0.15, GRIP_DOWN_QUAT_WXYZ))

    # Excessive tool tilt (pointing sideways/upward instead of downward)
    with pytest.raises(WorkspaceViolationError, match="tilt"):
        validator.validate_pose(Pose.from_xyz(0.30, 0.0, 0.15, (1.0, 0.0, 0.0, 0.0)))

    # Speed violations
    with pytest.raises(WorkspaceViolationError, match="exceeds limit"):
        validator.validate_speed(0.50)
    with pytest.raises(WorkspaceViolationError, match="must be positive"):
        validator.validate_speed(0.0)
