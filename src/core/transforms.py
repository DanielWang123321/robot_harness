"""Rigid-body SE(3) transforms, static calibration loaders, and XArmAPI TCP/Flange conversions."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from .types import Pose


@dataclass(frozen=True)
class HandEyeCalibration:
    """Eye-in-Hand calibration parameters loaded from camera_calibration.yaml."""

    serial: str
    width: int
    height: int
    K: np.ndarray               # (3, 3) intrinsic matrix
    distortion: np.ndarray      # (5,) distortion coefficients
    T_flange_cam: np.ndarray    # (4, 4) camera frame -> flange frame
    T_tcp_cam: np.ndarray       # (4, 4) camera frame -> TCP frame
    tcp_offset_z_m: float


@dataclass(frozen=True)
class KinematicsCalibration:
    """Static per-unit kinematic calibration loaded from xarm6_kinematics_*.yaml."""

    schema_version: int
    robot_key: str
    serial_number: str
    joints: dict[str, dict[str, float]]
    source_path: Path
    sha256: str


def quat_wxyz_to_matrix(quat_wxyz: np.ndarray | Sequence[float]) -> np.ndarray:
    """Convert quaternion (w, x, y, z) to 3x3 rotation matrix."""
    q = np.asarray(quat_wxyz, dtype=float).reshape(4)
    norm = float(np.linalg.norm(q))
    if norm < 1e-12:
        raise ValueError("Zero-norm quaternion is invalid")
    q = q / norm
    xyzw = np.array([q[1], q[2], q[3], q[0]], dtype=float)
    return Rotation.from_quat(xyzw).as_matrix()


def matrix_to_quat_wxyz(rot_mat: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to quaternion (w, x, y, z) with w >= 0."""
    R = np.asarray(rot_mat, dtype=float).reshape(3, 3)
    xyzw = Rotation.from_matrix(R).as_quat()
    wxyz = np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]], dtype=float)
    if wxyz[0] < 0:
        wxyz = -wxyz
    return wxyz


def rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Convert extrinsic XYZ roll-pitch-yaw (rad) to 3x3 rotation matrix."""
    return Rotation.from_euler("xyz", [roll, pitch, yaw], degrees=False).as_matrix()


def matrix_to_rpy(rot_mat: np.ndarray) -> tuple[float, float, float]:
    """Convert 3x3 rotation matrix to extrinsic XYZ roll-pitch-yaw (rad)."""
    rpy = Rotation.from_matrix(np.asarray(rot_mat, dtype=float).reshape(3, 3)).as_euler(
        "xyz", degrees=False
    )
    return float(rpy[0]), float(rpy[1]), float(rpy[2])


def rpy_to_quat_wxyz(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Convert extrinsic XYZ roll-pitch-yaw (rad) to quaternion [w, x, y, z]."""
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    quat = np.array(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ],
        dtype=np.float64,
    )
    return quat / np.linalg.norm(quat)


def quat_wxyz_to_rpy(quat: Sequence[float]) -> np.ndarray:
    """Convert quaternion [w, x, y, z] to extrinsic XYZ roll-pitch-yaw (rad)."""
    q = np.asarray(quat, dtype=np.float64).reshape(4)
    q = q / np.linalg.norm(q)
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (w * y - z * x)
    pitch = math.copysign(math.pi * 0.5, sinp) if abs(sinp) >= 1.0 else math.asin(sinp)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return np.array([roll, pitch, yaw], dtype=np.float64)


def pose_to_matrix(pose: Pose) -> np.ndarray:
    """Convert Pose to 4x4 homogeneous transformation matrix T."""
    T = np.eye(4, dtype=float)
    T[:3, :3] = quat_wxyz_to_matrix(pose.orientation_wxyz)
    T[:3, 3] = pose.pos_np
    return T


def matrix_to_pose(T: np.ndarray) -> Pose:
    """Convert 4x4 homogeneous transformation matrix T to Pose."""
    mat = np.asarray(T, dtype=float).reshape(4, 4)
    quat = matrix_to_quat_wxyz(mat[:3, :3])
    pos = mat[:3, 3]
    return Pose.from_xyz(float(pos[0]), float(pos[1]), float(pos[2]), quat)


def _pose_mm_rad_to_se3(pose_mm_rad: Sequence[float]) -> np.ndarray:
    arr = np.asarray(pose_mm_rad, dtype=np.float64).reshape(6)
    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = rpy_to_matrix(float(arr[3]), float(arr[4]), float(arr[5]))
    mat[:3, 3] = arr[:3]
    return mat


def _se3_to_pose_mm_rad(mat: np.ndarray) -> np.ndarray:
    r, p, y = matrix_to_rpy(mat[:3, :3])
    return np.array([mat[0, 3], mat[1, 3], mat[2, 3], r, p, y], dtype=np.float64)


def pose_flange_to_tcp(
    flange_pose_mm_rad: Sequence[float],
    tcp_offset_mm_rad: Sequence[float] | None = None,
) -> np.ndarray:
    """Transform flange pose [x_mm, y_mm, z_mm, r_rad, p_rad, y_rad] to controller TCP frame."""
    if tcp_offset_mm_rad is None:
        return np.asarray(flange_pose_mm_rad, dtype=np.float64).reshape(6).copy()
    t_flange = _pose_mm_rad_to_se3(flange_pose_mm_rad)
    t_offset = _pose_mm_rad_to_se3(tcp_offset_mm_rad)
    return _se3_to_pose_mm_rad(t_flange @ t_offset)


def pose_tcp_to_flange(
    tcp_pose_mm_rad: Sequence[float],
    tcp_offset_mm_rad: Sequence[float] | None = None,
) -> np.ndarray:
    """Transform controller TCP pose [x_mm, y_mm, z_mm, r_rad, p_rad, y_rad] back to flange frame."""
    if tcp_offset_mm_rad is None:
        return np.asarray(tcp_pose_mm_rad, dtype=np.float64).reshape(6).copy()
    t_tcp = _pose_mm_rad_to_se3(tcp_pose_mm_rad)
    t_offset = _pose_mm_rad_to_se3(tcp_offset_mm_rad)
    t_offset_inv = np.eye(4, dtype=np.float64)
    r_T = t_offset[:3, :3].T
    t_offset_inv[:3, :3] = r_T
    t_offset_inv[:3, 3] = -r_T @ t_offset[:3, 3]
    return _se3_to_pose_mm_rad(t_tcp @ t_offset_inv)


def flange_m_quat_to_tcp_mm_rpy(
    pos_m: Sequence[float],
    quat_wxyz: Sequence[float],
    tcp_offset_mm_rad: Sequence[float] | None = None,
) -> np.ndarray:
    """Convert MuJoCo flange (pos_m, quat_wxyz) to controller TCP [x_mm, y_mm, z_mm, r_rad, p_rad, y_rad]."""
    pos_mm = np.asarray(pos_m, dtype=np.float64).reshape(3) * 1000.0
    rpy = quat_wxyz_to_rpy(quat_wxyz)
    flange = np.concatenate([pos_mm, rpy])
    return pose_flange_to_tcp(flange, tcp_offset_mm_rad)


def tcp_mm_rpy_to_flange_m_quat(
    tcp_pose_mm_rad: Sequence[float],
    tcp_offset_mm_rad: Sequence[float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert controller TCP [x_mm, y_mm, z_mm, r_rad, p_rad, y_rad] to flange (pos_m, quat_wxyz)."""
    flange = pose_tcp_to_flange(tcp_pose_mm_rad, tcp_offset_mm_rad)
    pos_m = flange[:3] / 1000.0
    quat = rpy_to_quat_wxyz(float(flange[3]), float(flange[4]), float(flange[5]))
    return pos_m, quat


def pixel_depth_to_camera_point(
    u: float,
    v: float,
    depth_m: float,
    K: np.ndarray,
) -> np.ndarray:
    """Back-project pixel (u, v) and depth (m) to 3D point in camera optical frame."""
    if depth_m <= 0.0 or not np.isfinite(depth_m):
        raise ValueError(f"Depth must be positive and finite, got {depth_m}")
    K_mat = np.asarray(K, dtype=float).reshape(3, 3)
    fx, fy = float(K_mat[0, 0]), float(K_mat[1, 1])
    cx, cy = float(K_mat[0, 2]), float(K_mat[1, 2])
    x = (float(u) - cx) * float(depth_m) / fx
    y = (float(v) - cy) * float(depth_m) / fy
    z = float(depth_m)
    return np.array([x, y, z], dtype=float)


def camera_point_to_base(
    point_cam: np.ndarray,
    T_base_flange: np.ndarray,
    T_flange_cam: np.ndarray,
) -> np.ndarray:
    """Transform a 3D point from camera frame to robot base frame:
    P_base = T_base_flange @ T_flange_cam @ P_cam.
    """
    p_cam = np.asarray(point_cam, dtype=float).reshape(3)
    p_hom = np.array([p_cam[0], p_cam[1], p_cam[2], 1.0], dtype=float)
    T_bf = np.asarray(T_base_flange, dtype=float).reshape(4, 4)
    T_fc = np.asarray(T_flange_cam, dtype=float).reshape(4, 4)
    p_base_hom = T_bf @ T_fc @ p_hom
    return p_base_hom[:3].copy()


def load_hand_eye_calibration(path: str | Path) -> HandEyeCalibration:
    """Load static Eye-in-Hand calibration YAML."""
    yaml_path = Path(path)
    raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    cam = raw["camera"]
    ext = raw["extrinsics"]
    tcp_offset_mm = ext.get("tcp_offset_mm_rad", [0.0, 0.0, 172.0, 0.0, 0.0, 0.0])
    return HandEyeCalibration(
        serial=str(cam.get("serial", "")),
        width=int(cam["width"]),
        height=int(cam["height"]),
        K=np.array(cam["K"], dtype=float).reshape(3, 3),
        distortion=np.array(cam["distortion"], dtype=float).reshape(-1),
        T_flange_cam=np.array(ext["X_flange_cam"], dtype=float).reshape(4, 4),
        T_tcp_cam=np.array(ext["X_tcp_cam"], dtype=float).reshape(4, 4),
        tcp_offset_z_m=float(tcp_offset_mm[2]) / 1000.0,
    )


def verify_calibration_orthonormality(calib: HandEyeCalibration, tol: float = 1e-5) -> bool:
    """Verify that T_flange_cam and T_tcp_cam have valid SO(3) rotation blocks."""
    for mat in (calib.T_flange_cam, calib.T_tcp_cam):
        R = mat[:3, :3]
        err_orth = float(np.max(np.abs(R.T @ R - np.eye(3))))
        det_err = abs(float(np.linalg.det(R)) - 1.0)
        if err_orth > tol or det_err > tol:
            return False
    return True


def load_kinematics_calibration(yaml_path: str | Path) -> KinematicsCalibration:
    """Load static per-unit kinematics calibration YAML (e.g. configs/xarm6_kinematics_L56A01.yaml)."""
    path = Path(yaml_path).expanduser().resolve()
    raw_bytes = path.read_bytes()
    data: dict[str, Any] = yaml.safe_load(raw_bytes.decode("utf-8"))
    raw_key = str(data.get("robot_key", "xarm6")).strip()
    robot_key = "xarm6" if raw_key.startswith("xarm6") else raw_key
    joints_raw = data["joints"]
    joints: dict[str, dict[str, float]] = {}
    for jname, axes in joints_raw.items():
        joints[str(jname)] = {str(k): float(v) for k, v in axes.items()}
    return KinematicsCalibration(
        schema_version=int(data.get("schema_version", 1)),
        robot_key=robot_key,
        serial_number=str(data.get("serial_number", "NOMINAL")).strip(),
        joints=joints,
        source_path=path,
        sha256=hashlib.sha256(raw_bytes).hexdigest(),
    )


def apply_calibration_to_model(
    model: mujoco.MjModel,
    calibration: KinematicsCalibration,
    data: mujoco.MjData | None = None,
) -> None:
    """Patch a MuJoCo model in-place with static per-unit joint origin calibration."""
    for joint_name, params in calibration.joints.items():
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            continue
        body_id = int(model.jnt_bodyid[joint_id])
        model.body_pos[body_id] = (params["x"], params["y"], params["z"])
        model.body_quat[body_id] = rpy_to_quat_wxyz(
            params["roll"], params["pitch"], params["yaw"]
        )
    target_data = data if data is not None else mujoco.MjData(model)
    mujoco.mj_setConst(model, target_data)
    mujoco.mj_forward(model, target_data)
