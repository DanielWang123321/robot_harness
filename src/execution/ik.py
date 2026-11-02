"""Damped least-squares (DLS) inverse kinematics and Gripper G2 linkage kinematics."""

from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import numpy as np

GRIPPER_G2_CLOSED_GAP_MM = 0.0
GRIPPER_G2_OPEN_GAP_MM = 84.0
GRIPPER_G2_OPEN_GAP_M = 0.084
GRIPPER_G2_SIM_OPEN_DRIVE = 0.057210812728026743
GRIPPER_G2_SIM_CLOSE_DRIVE = 0.86509464599328212


@dataclass(frozen=True)
class G2LinkageGeometry:
    """Parallel-pad 4-bar linkage dimensions for UFACTORY Gripper G2."""

    base_joint_y_m: float = 0.035
    finger_joint_y_m: float = 0.035465
    finger_joint_z_m: float = 0.042039
    pad_inner_y_m: float = -0.02600320242345333
    pad_center_z_m: float = 0.041003202088177204
    drive_lower_rad: float = GRIPPER_G2_SIM_OPEN_DRIVE
    drive_upper_rad: float = GRIPPER_G2_SIM_CLOSE_DRIVE

    @property
    def radius_m(self) -> float:
        return math.hypot(self.finger_joint_y_m, self.finger_joint_z_m)

    @property
    def phase_rad(self) -> float:
        return math.atan2(self.finger_joint_z_m, self.finger_joint_y_m)

    def pad_gap_m(self, drive_rad: float) -> float:
        q = float(drive_rad)
        return 2.0 * (
            self.base_joint_y_m
            + self.finger_joint_y_m * math.cos(q)
            - self.finger_joint_z_m * math.sin(q)
            + self.pad_inner_y_m
        )

    def drive_for_gap_m(self, gap_m: float) -> float:
        gap = max(0.0, min(GRIPPER_G2_OPEN_GAP_M, float(gap_m)))
        cosine = (gap / 2.0 - self.base_joint_y_m - self.pad_inner_y_m) / self.radius_m
        q = math.acos(max(-1.0, min(1.0, cosine))) - self.phase_rad
        return max(self.drive_lower_rad, min(self.drive_upper_rad, q))


G2_LINKAGE_GEOMETRY = G2LinkageGeometry()


def gripper_g2_gap_m_to_sim_drive(gap_m: float) -> float:
    return G2_LINKAGE_GEOMETRY.drive_for_gap_m(gap_m)


def gripper_g2_sim_drive_to_gap_m(drive_rad: float) -> float:
    clipped = max(GRIPPER_G2_SIM_OPEN_DRIVE, min(GRIPPER_G2_SIM_CLOSE_DRIVE, float(drive_rad)))
    return max(0.0, min(GRIPPER_G2_OPEN_GAP_M, G2_LINKAGE_GEOMETRY.pad_gap_m(clipped)))


def sdk_position_counts(position_mm: float) -> int:
    pos = max(0.0, min(84.0, float(position_mm)))
    return int((math.degrees(math.asin((pos - 16.0) / 110.0)) + 8.33) * 18.28)


def sdk_position_mm(raw_position: float) -> float:
    return math.sin(math.radians(float(raw_position) / 18.28 - 8.33)) * 110.0 + 16.0


def quat_error(quat_cur: np.ndarray, quat_target: np.ndarray) -> np.ndarray:
    """Orientation error as world-frame rotation vector: e = target * conj(cur)."""
    q_conj = np.array([quat_cur[0], -quat_cur[1], -quat_cur[2], -quat_cur[3]], dtype=float)
    q_err = np.empty(4, dtype=float)
    mujoco.mju_mulQuat(q_err, np.asarray(quat_target, dtype=float), q_conj)
    rot = np.empty(3, dtype=float)
    mujoco.mju_quat2Vel(rot, q_err, 1.0)
    return rot


def quintic_interpolate(t: np.ndarray, q0: np.ndarray, q1: np.ndarray, duration: float) -> np.ndarray:
    """Zero velocity/acceleration quintic polynomial interpolation for t in [0, duration]."""
    s = np.clip(t / max(duration, 1e-6), 0.0, 1.0)
    s5 = 10.0 * s**3 - 15.0 * s**4 + 6.0 * s**5
    return q0 + (q1 - q0) * s5[:, None]


def solve_ik(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    site_name: str,
    target_pos: np.ndarray,
    target_quat: np.ndarray | None,
    q_init: np.ndarray | None = None,
    *,
    pos_tol: float = 1e-4,
    rot_tol: float = 1e-3,
    max_iter: int = 100,
    damping: float = 1e-3,
) -> tuple[np.ndarray, bool]:
    """Solve Damped Least-Squares (DLS) IK for the named site. Returns (q, converged)."""
    site_id = model.site(site_name).id
    q = data.qpos.copy() if q_init is None else np.asarray(q_init, dtype=float).copy()
    target_p = np.asarray(target_pos, dtype=float).reshape(3)
    target_q = None if target_quat is None else np.asarray(target_quat, dtype=float).reshape(4)
    tol_vec = np.array([pos_tol] * 3 + ([rot_tol] * 3 if target_q is not None else []), dtype=float)

    jacp = np.zeros((3, model.nv), dtype=float)
    jacr = np.zeros((3, model.nv), dtype=float)

    jnt_types = model.jnt_type
    clamp_mask = (jnt_types == int(mujoco.mjtJoint.mjJNT_HINGE)) | (
        jnt_types == int(mujoco.mjtJoint.mjJNT_SLIDE)
    )
    clamp_adrs = model.jnt_qposadr[clamp_mask]
    clamp_lo = model.jnt_range[clamp_mask, 0]
    clamp_hi = model.jnt_range[clamp_mask, 1]
    has_clamp = clamp_adrs.size > 0
    quat = np.empty(4, dtype=float)
    quat_trial = np.empty(4, dtype=float)
    eye_m = (damping**2) * np.eye(6 if target_q is not None else 3, dtype=float)

    def _clamp(qvec: np.ndarray) -> None:
        if has_clamp:
            qvec[clamp_adrs] = np.clip(qvec[clamp_adrs], clamp_lo, clamp_hi)

    def _violation(scaled_err: np.ndarray) -> float:
        return float(np.sum(np.maximum(np.abs(scaled_err) - 1.0, 0.0)))

    step = 0.5
    for _ in range(max_iter):
        data.qpos[:] = q
        mujoco.mj_kinematics(model, data)

        pos_err = target_p - data.site_xpos[site_id]
        err = pos_err
        if target_q is not None:
            mujoco.mju_mat2Quat(quat, data.site_xmat[site_id])
            rot_err = quat_error(quat, target_q)
            err = np.concatenate([pos_err, rot_err])
            if np.all(np.abs(err) < tol_vec):
                break
        elif float(np.linalg.norm(pos_err)) < pos_tol:
            break

        err_scaled = err / tol_vec
        mujoco.mj_comPos(model, data)
        mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
        J = jacp if target_q is None else np.vstack([jacp, jacr])
        JJt = J @ J.T + eye_m
        dq = J.T @ np.linalg.solve(JJt, err)

        v_cur = _violation(err_scaled)
        accepted = False
        for _ in range(8):
            q_trial = q.copy()
            mujoco.mj_integratePos(model, q_trial, dq, step)
            _clamp(q_trial)
            data.qpos[:] = q_trial
            mujoco.mj_kinematics(model, data)
            pos_trial = target_p - data.site_xpos[site_id]
            if target_q is not None:
                mujoco.mju_mat2Quat(quat_trial, data.site_xmat[site_id])
                err_trial = np.concatenate([pos_trial, quat_error(quat_trial, target_q)])
            else:
                err_trial = pos_trial
            if _violation(err_trial / tol_vec) < v_cur:
                q = q_trial
                accepted = True
                break
            step *= 0.5
        if not accepted:
            break
        step = min(step * 1.5, 0.5)

    data.qpos[:] = q
    mujoco.mj_kinematics(model, data)
    mujoco.mj_comPos(model, data)
    pos_err_vec = target_p - data.site_xpos[site_id]
    converged = bool(np.all(np.abs(pos_err_vec) < pos_tol))
    if target_q is not None:
        mujoco.mju_mat2Quat(quat, data.site_xmat[site_id])
        rot_err_vec = quat_error(quat, target_q)
        converged = converged and bool(np.all(np.abs(rot_err_vec) < rot_tol))
    return q.copy(), converged


def solve_ik_trajectory(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    site_name: str,
    waypoints_pos: np.ndarray,
    quat: np.ndarray | None,
    q_start: np.ndarray,
    *,
    pos_tol: float = 2e-4,
) -> tuple[np.ndarray, bool]:
    """Sequential IK along Cartesian waypoints. Returns (q_traj (N, nq), all_converged)."""
    qs = np.empty((len(waypoints_pos), model.nq), dtype=float)
    ok = True
    q = np.asarray(q_start, dtype=float).copy()
    quat_arr = None if quat is None else np.asarray(quat, dtype=float)
    if quat_arr is not None and quat_arr.ndim == 1:
        quat_arr = np.broadcast_to(quat_arr, (len(waypoints_pos), 4))
    for i, p in enumerate(waypoints_pos):
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        q, conv = solve_ik(
            model,
            data,
            site_name,
            p,
            None if quat_arr is None else quat_arr[i],
            q_init=q,
            pos_tol=pos_tol,
        )
        qs[i] = q
        ok &= conv
    return qs, ok
