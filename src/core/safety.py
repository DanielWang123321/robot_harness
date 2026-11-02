"""Workspace boundary validator, MuJoCo collision checker, and XArmAPI error codes."""

from __future__ import annotations

import math
from pathlib import Path

import mujoco
import numpy as np

from .transforms import quat_wxyz_to_matrix
from .types import Pose, WorkspaceLimits

# Standard XArmAPI return codes
XARM_OK = 0
XARM_ERR_NOT_READY = 1
XARM_ERR_EMERGENCY_STOP = 2
XARM_ERR_IK_FAILED = 21
XARM_ERR_OUT_OF_RANGE = 31
XARM_ERR_COLLISION = 32
XARM_ERR_READ_ONLY = 33
XARM_ERR_INVALID_PARAM = 34

_GRIPPER_ACTUATOR_NAMES = ("gripper", "gripper_motor", "finger_actuator")


class WorkspaceViolationError(ValueError):
    """Raised when a commanded pose or velocity violates workspace safety limits."""


class WorkspaceValidator:
    """Validates Cartesian waypoints and orientations before execution."""

    def __init__(self, limits: WorkspaceLimits | None = None) -> None:
        self.limits = limits or WorkspaceLimits()

    def validate_pose(self, pose: Pose, *, label: str = "target") -> None:
        x, y, z = pose.position
        lim = self.limits

        if not all(math.isfinite(v) for v in (x, y, z)):
            raise WorkspaceViolationError(f"{label} position contains non-finite values: {pose.position}")

        if not (lim.x_min_m <= x <= lim.x_max_m):
            raise WorkspaceViolationError(
                f"{label} X={x:.4f} m outside [{lim.x_min_m:.4f}, {lim.x_max_m:.4f}] m"
            )
        if not (lim.y_min_m <= y <= lim.y_max_m):
            raise WorkspaceViolationError(
                f"{label} Y={y:.4f} m outside [{lim.y_min_m:.4f}, {lim.y_max_m:.4f}] m"
            )
        if not (lim.z_min_m <= z <= lim.z_max_m):
            raise WorkspaceViolationError(
                f"{label} Z={z:.4f} m outside [{lim.z_min_m:.4f}, {lim.z_max_m:.4f}] m (table collision guard)"
            )

        rot = quat_wxyz_to_matrix(pose.orientation_wxyz)
        tool_z = rot[:, 2]
        # Tool Z-axis should point downward: (0, 0, -1)
        cos_tilt = float(np.clip(np.dot(tool_z, np.array([0.0, 0.0, -1.0])), -1.0, 1.0))
        tilt_deg = math.degrees(math.acos(cos_tilt))
        if tilt_deg > lim.max_tilt_deg:
            raise WorkspaceViolationError(
                f"{label} tool tilt {tilt_deg:.2f} deg exceeds max_tilt_deg={lim.max_tilt_deg:.2f} deg"
            )

    def validate_speed(self, speed_m_s: float) -> float:
        if not math.isfinite(speed_m_s) or speed_m_s <= 0.0:
            raise WorkspaceViolationError(f"Cartesian speed must be positive, got {speed_m_s}")
        if speed_m_s > self.limits.max_cart_vel_m_s:
            raise WorkspaceViolationError(
                f"Cartesian speed {speed_m_s:.3f} m/s exceeds limit {self.limits.max_cart_vel_m_s:.3f} m/s"
            )
        return float(speed_m_s)


def arm_joint_limits(model: mujoco.MjModel) -> tuple[int, np.ndarray, np.ndarray]:
    """Return (n_arm_qpos, q_lo, q_hi) for the arm joints."""
    n_arm_actuators = model.nu
    for i in range(model.nu):
        if model.actuator(i).name in _GRIPPER_ACTUATOR_NAMES:
            n_arm_actuators = i
            break
    joints: list[int] = []
    for i in range(n_arm_actuators):
        if int(model.actuator_trntype[i]) == int(mujoco.mjtTrn.mjTRN_JOINT):
            joints.append(int(model.actuator_trnid[i, 0]))
    if not joints:
        raise ValueError("model has no arm actuators driving joints")
    joints.sort(key=lambda j: int(model.jnt_qposadr[j]))
    qpos_adr = [int(model.jnt_qposadr[j]) for j in joints]
    if max(qpos_adr) + 1 != len(qpos_adr):
        raise ValueError("arm joint qpos entries are not the leading qpos block")
    lo = np.array([model.jnt_range[j, 0] for j in joints], dtype=float)
    hi = np.array([model.jnt_range[j, 1] for j in joints], dtype=float)
    return len(qpos_adr), lo, hi


class CollisionChecker:
    """Arm-joint collision and joint-limit queries against a private copy of a scene model."""

    def __init__(
        self,
        scene: str | Path,
        *,
        safety_margin: float = 0.0,
        query_margin: float = 0.02,
        ignore_pairs: tuple[tuple[str, str], ...] = (),
        ignore_bodies: tuple[str, ...] = ("world", "cube"),
    ) -> None:
        self.model = mujoco.MjModel.from_xml_path(str(scene))
        self.data = mujoco.MjData(self.model)
        self.safety_margin = float(safety_margin)
        self.query_margin = float(query_margin)

        self.n_arm_qpos, self.q_lo, self.q_hi = arm_joint_limits(self.model)

        active = self.model.geom_contype.astype(bool) | self.model.geom_conaffinity.astype(bool)
        self.model.geom_margin[active] = self.query_margin

        self._ignore: set[frozenset[int]] = set()
        for b1, b2 in ignore_pairs:
            id1 = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, b1)
            id2 = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, b2)
            if id1 >= 0 and id2 >= 0:
                self._ignore.add(frozenset((id1, id2)))

        self._ignore_bodies: set[int] = set()
        for name in ignore_bodies:
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            if body_id >= 0:
                self._ignore_bodies.add(body_id)

    def within_joint_limits(self, q: np.ndarray, tol: float = 1e-6) -> bool:
        arr = np.asarray(q, dtype=float).reshape(self.n_arm_qpos)
        return bool(np.all(arr >= self.q_lo - tol) and np.all(arr <= self.q_hi + tol))

    def _contacts(self, q: np.ndarray) -> list[tuple[int, int, float]]:
        arr = np.asarray(q, dtype=float).reshape(self.n_arm_qpos)
        self.data.qpos[: self.n_arm_qpos] = arr
        mujoco.mj_forward(self.model, self.data)
        out: list[tuple[int, int, float]] = []
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            b1 = int(self.model.geom_bodyid[c.geom[0]])
            b2 = int(self.model.geom_bodyid[c.geom[1]])
            if b1 in self._ignore_bodies or b2 in self._ignore_bodies:
                continue
            if frozenset((b1, b2)) in self._ignore:
                continue
            out.append((b1, b2, float(c.dist)))
        return out

    def in_collision(self, q: np.ndarray, *, safety_margin: float | None = None) -> bool:
        margin = self.safety_margin if safety_margin is None else float(safety_margin)
        return any(dist < margin for _, _, dist in self._contacts(q))

    def min_distance(self, q: np.ndarray) -> float:
        contacts = self._contacts(q)
        return min((dist for _, _, dist in contacts), default=float("inf"))
