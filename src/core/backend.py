"""Unified RobotBackend abstract interface aligned with xarm-python-sdk (XArmAPI) signatures."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np

from .safety import XARM_OK
from .transforms import matrix_to_rpy, quat_wxyz_to_matrix
from .types import Pose, RobotState


class RobotBackend(ABC):
    """Abstract robot execution interface aligned with XArmAPI for zero-friction Sim-to-Real."""

    # ------------------------------------------------------------------
    # Standard XArmAPI System & State Methods
    # ------------------------------------------------------------------
    @abstractmethod
    def motion_enable(self, enable: bool = True, servo_id: int | None = None) -> int:
        """Enable or disable motor motion (XArmAPI compatible). Returns code (0 = OK)."""

    @abstractmethod
    def set_mode(self, mode: int = 0) -> int:
        """Set controller operating mode (0 = position control). Returns code (0 = OK)."""

    @abstractmethod
    def set_state(self, state: int = 0) -> int:
        """Set controller state (0 = sport/ready, 3 = pause, 4 = stop). Returns code (0 = OK)."""

    @abstractmethod
    def clean_error(self) -> int:
        """Clear controller error code. Returns code (0 = OK)."""

    @abstractmethod
    def clean_warn(self) -> int:
        """Clear controller warning code. Returns code (0 = OK)."""

    @abstractmethod
    def emergency_stop(self) -> None:
        """Immediately stop all motion (XArmAPI compatible)."""

    @abstractmethod
    def get_state(self) -> tuple[int, int]:
        """Return (code, state) following XArmAPI convention (1=in motion, 2=standby, 4=stop)."""

    @abstractmethod
    def get_err_warn_code(self) -> tuple[int, list[int]]:
        """Return (code, [error_code, warn_code]) following XArmAPI convention."""

    @abstractmethod
    def get_robot_sn(self) -> tuple[int, str]:
        """Return (code, serial_number) following XArmAPI convention."""

    # ------------------------------------------------------------------
    # Standard XArmAPI Kinematics & Motion Methods
    # ------------------------------------------------------------------
    @abstractmethod
    def set_tcp_offset(self, offset: Sequence[float], is_radian: bool = False) -> int:
        """Set tool center point offset [x_mm, y_mm, z_mm, roll, pitch, yaw]. Returns code (0 = OK)."""

    @abstractmethod
    def get_position(self, is_radian: bool = False) -> tuple[int, list[float]]:
        """Return (code, [x_mm, y_mm, z_mm, roll, pitch, yaw]) of current TCP pose."""

    @abstractmethod
    def get_servo_angle(
        self,
        servo_id: int | None = None,
        is_radian: bool = False,
    ) -> tuple[int, list[float] | float]:
        """Return (code, [j1..j6]) or single joint angle if servo_id (1..6) is specified."""

    @abstractmethod
    def set_position(
        self,
        x: float | None = None,
        y: float | None = None,
        z: float | None = None,
        roll: float | None = None,
        pitch: float | None = None,
        yaw: float | None = None,
        radius: float | None = None,
        speed: float | None = None,
        mvacc: float | None = None,
        mvtime: float | None = None,
        relative: bool = False,
        is_radian: bool = False,
        wait: bool = True,
    ) -> int:
        """Move TCP in Cartesian space (mm, deg/rad). Unspecified coordinates keep current values."""

    @abstractmethod
    def set_servo_angle(
        self,
        servo_id: int | None = None,
        angle: Sequence[float] | float | None = None,
        speed: float | None = None,
        mvacc: float | None = None,
        mvtime: float | None = None,
        relative: bool = False,
        is_radian: bool = False,
        wait: bool = True,
    ) -> int:
        """Move joints in joint space (deg/rad). Returns code (0 = OK)."""

    @abstractmethod
    def reset(
        self,
        speed: float | None = None,
        mvacc: float | None = None,
        mvtime: float | None = None,
        is_radian: bool = False,
        wait: bool = True,
    ) -> int:
        """Move arm back to home/zero configuration (XArmAPI compatible). Returns code (0 = OK)."""

    # ------------------------------------------------------------------
    # Standard XArmAPI Gripper & Gripper G2 Methods
    # ------------------------------------------------------------------
    @abstractmethod
    def set_gripper_enable(self, enable: bool = True) -> int:
        """Enable or disable the gripper. Returns code (0 = OK)."""

    @abstractmethod
    def set_gripper_mode(self, mode: int = 0) -> int:
        """Set gripper mode (0 = location mode). Returns code (0 = OK)."""

    @abstractmethod
    def set_gripper_speed(self, speed: float) -> int:
        """Set gripper speed in pulse/s (r/min). Returns code (0 = OK)."""

    @abstractmethod
    def get_gripper_position(self) -> tuple[int, float]:
        """Return (code, pos_counts) in SDK pulse scale [0..840]."""

    @abstractmethod
    def set_gripper_position(
        self,
        pos: float,
        wait: bool = True,
        speed: float | None = None,
    ) -> int:
        """Command gripper position in SDK pulse scale [0..840]. Returns code (0 = OK)."""

    @abstractmethod
    def set_gripper_g2_position(
        self,
        pos: int,
        speed: int = 100,
        force: int = 50,
        wait: bool = True,
    ) -> int:
        """Command Gripper G2 physical gap in mm [0..84] with speed (mm/s) and force (%)."""

    # ------------------------------------------------------------------
    # Harness Structured Telemetry & High-Level Convenience Wrappers
    # ------------------------------------------------------------------
    @abstractmethod
    def get_robot_state(self) -> RobotState:
        """Return structured RobotState (SI units + contact forces) for Perception and Harness."""

    def move_home(self) -> None:
        """High-level wrapper over XArmAPI reset()."""
        code = self.reset(wait=True)
        if code != XARM_OK:
            raise RuntimeError(f"reset/move_home failed with XArmAPI code={code}")

    def move_tcp(self, pose: Pose, speed_m_s: float = 0.15) -> None:
        """High-level SI wrapper over XArmAPI set_position()."""
        xyz_mm = np.asarray(pose.position, dtype=float) * 1000.0
        r_rad, p_rad, y_rad = matrix_to_rpy(quat_wxyz_to_matrix(pose.orientation_wxyz))
        code = self.set_position(
            x=float(xyz_mm[0]),
            y=float(xyz_mm[1]),
            z=float(xyz_mm[2]),
            roll=math.degrees(r_rad),
            pitch=math.degrees(p_rad),
            yaw=math.degrees(y_rad),
            speed=float(speed_m_s * 1000.0),
            is_radian=False,
            wait=True,
        )
        if code != XARM_OK:
            raise RuntimeError(f"set_position failed with XArmAPI code={code} for pose={pose}")

    def open_gripper(self) -> None:
        """High-level wrapper over XArmAPI set_gripper_g2_position(84)."""
        code = self.set_gripper_g2_position(pos=84, speed=150, force=50, wait=True)
        if code != XARM_OK:
            raise RuntimeError(f"open_gripper failed with XArmAPI code={code}")

    def close_gripper(self, target_gap_m: float = 0.022, target_body: str | None = None) -> float:
        """High-level wrapper over XArmAPI set_gripper_g2_position()."""
        del target_body
        gap_mm = round(max(0.0, min(84.0, float(target_gap_m) * 1000.0)))
        code = self.set_gripper_g2_position(pos=gap_mm, speed=120, force=70, wait=True)
        if code != XARM_OK:
            raise RuntimeError(f"close_gripper failed with XArmAPI code={code}")
        return self.get_robot_state().gripper_gap_m
