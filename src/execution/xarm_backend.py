"""Real UFACTORY xArm 6 + Gripper G2 hardware backend implementing XArmAPI-compatible RobotBackend."""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from typing import Any

import numpy as np

from core.backend import RobotBackend
from core.safety import (
    XARM_ERR_INVALID_PARAM,
    XARM_ERR_OUT_OF_RANGE,
    XARM_OK,
    WorkspaceValidator,
    WorkspaceViolationError,
)
from core.transforms import (
    flange_m_quat_to_tcp_mm_rpy,
    matrix_to_pose,
    matrix_to_rpy,
    rpy_to_matrix,
    rpy_to_quat_wxyz,
    tcp_mm_rpy_to_flange_m_quat,
)
from core.types import Pose, RobotState, WorkspaceLimits

from .ik import GRIPPER_G2_OPEN_GAP_M, sdk_position_counts, sdk_position_mm


class XArmSafetyError(RuntimeError):
    """Raised when hardware safety gate or controller state check fails."""


class XArmBackend(RobotBackend):
    """Real xArm 6 hardware backend (`192.168.1.60`) with XArmAPI signatures and read_only gate."""

    def __init__(
        self,
        ip: str = "192.168.1.60",
        *,
        read_only: bool = True,
        tcp_offset_z_m: float = 0.1735,
        home_joints_rad: tuple[float, ...] = (0.0, -0.6315299, -0.8331505, 0.0, 1.4646804, 0.0),
        limits: WorkspaceLimits | None = None,
        arm_api: Any | None = None,
    ) -> None:
        self.ip = str(ip)
        self.read_only = bool(read_only)
        self.tcp_offset_z_m = float(tcp_offset_z_m)
        self._tcp_offset_mm_rad = np.array(
            [0.0, 0.0, float(tcp_offset_z_m) * 1000.0, 0.0, 0.0, 0.0],
            dtype=float,
        )
        self.home_joints_rad = tuple(float(q) for q in home_joints_rad)
        self.validator = WorkspaceValidator(limits)
        self._owns_connection = arm_api is None

        if arm_api is not None:
            self._arm = arm_api
        else:
            from xarm.wrapper import XArmAPI

            self._arm = XArmAPI(self.ip, is_radian=True, do_not_open=False)
            time.sleep(0.25)

        if not bool(getattr(self._arm, "connected", False)):
            raise XArmSafetyError(f"Cannot connect to xArm 6 at {self.ip}")

        code, err_warn = self._arm.get_err_warn_code()
        self.last_error_code = int(err_warn[0]) if code == 0 else -1
        self.last_warn_code = int(err_warn[1]) if code == 0 else -1
        if not self.read_only and self.last_error_code != 0:
            raise XArmSafetyError(
                f"xArm controller reports error_code={self.last_error_code} warn_code={self.last_warn_code}"
            )

    def disconnect(self) -> None:
        if self._owns_connection and getattr(self._arm, "connected", False):
            self._arm.disconnect()

    def close(self) -> None:
        self.disconnect()

    def _require_writable(self, action_name: str) -> None:
        if self.read_only:
            raise XArmSafetyError(
                f"XArmBackend is in read_only=True mode; blocked hardware motion command: {action_name}"
            )

    def _prepare_motion(self) -> None:
        self._require_writable("prepare_motion")
        code, err_warn = self._arm.get_err_warn_code()
        if code == 0 and err_warn[0] != 0:
            raise XArmSafetyError(
                f"Cannot enable motion while controller error_code={err_warn[0]} (e.g. E-Stop pressed)"
            )
        self._arm.clean_error()
        self._arm.clean_warn()
        self._arm.motion_enable(enable=True)
        self._arm.set_mode(0)
        self._arm.set_state(0)
        time.sleep(0.1)

    # ------------------------------------------------------------------
    # Standard XArmAPI System & State Methods
    # ------------------------------------------------------------------
    def motion_enable(self, enable: bool = True, servo_id: int | None = None) -> int:
        self._require_writable("motion_enable")
        return int(self._arm.motion_enable(enable=enable, servo_id=servo_id))

    def set_mode(self, mode: int = 0) -> int:
        self._require_writable("set_mode")
        return int(self._arm.set_mode(int(mode)))

    def set_state(self, state: int = 0) -> int:
        self._require_writable("set_state")
        return int(self._arm.set_state(int(state)))

    def clean_error(self) -> int:
        self._require_writable("clean_error")
        return int(self._arm.clean_error())

    def clean_warn(self) -> int:
        self._require_writable("clean_warn")
        return int(self._arm.clean_warn())

    def emergency_stop(self) -> None:
        self._arm.emergency_stop()

    def get_state(self) -> tuple[int, int]:
        code, state = self._arm.get_state()
        return int(code), int(state)

    def get_err_warn_code(self) -> tuple[int, list[int]]:
        code, err_warn = self._arm.get_err_warn_code()
        if int(code) != 0:
            return int(code), [-1, -1]
        return 0, [int(err_warn[0]), int(err_warn[1])]

    def get_robot_sn(self) -> tuple[int, str]:
        code, sn = self._arm.get_robot_sn()
        return int(code), str(sn or "").strip().rstrip("\x00")

    def get_serial_number(self) -> str:
        code, sn = self.get_robot_sn()
        if code != 0:
            raise XArmSafetyError(f"get_robot_sn failed with code={code}")
        return sn

    # ------------------------------------------------------------------
    # Standard XArmAPI Kinematics & Motion Methods
    # ------------------------------------------------------------------
    def set_tcp_offset(self, offset: Sequence[float], is_radian: bool = False) -> int:
        arr = np.asarray(offset, dtype=float).reshape(-1)
        if arr.shape != (6,) or not np.all(np.isfinite(arr)):
            return XARM_ERR_INVALID_PARAM
        rpy_rad = arr[3:6] if is_radian else np.deg2rad(arr[3:6])
        self._tcp_offset_mm_rad = np.array(
            [arr[0], arr[1], arr[2], rpy_rad[0], rpy_rad[1], rpy_rad[2]],
            dtype=float,
        )
        self.tcp_offset_z_m = float(arr[2]) / 1000.0
        return XARM_OK

    def _get_flange_pose_m_quat(self) -> Pose:
        code_p, pos_rpy = self._arm.get_position(is_radian=True)
        if code_p != 0:
            raise XArmSafetyError(f"get_position failed with code={code_p}")
        T_base_flange = np.eye(4, dtype=float)
        T_base_flange[:3, 3] = np.array(pos_rpy[:3], dtype=float) / 1000.0
        T_base_flange[:3, :3] = rpy_to_matrix(float(pos_rpy[3]), float(pos_rpy[4]), float(pos_rpy[5]))
        return matrix_to_pose(T_base_flange)

    def get_position(self, is_radian: bool = False) -> tuple[int, list[float]]:
        flange = self._get_flange_pose_m_quat()
        tcp_mm_rad = flange_m_quat_to_tcp_mm_rpy(
            flange.position, flange.orientation_wxyz, self._tcp_offset_mm_rad
        )
        if is_radian:
            return XARM_OK, [float(v) for v in tcp_mm_rad]
        return XARM_OK, [
            float(tcp_mm_rad[0]),
            float(tcp_mm_rad[1]),
            float(tcp_mm_rad[2]),
            float(math.degrees(tcp_mm_rad[3])),
            float(math.degrees(tcp_mm_rad[4])),
            float(math.degrees(tcp_mm_rad[5])),
        ]

    def get_servo_angle(
        self,
        servo_id: int | None = None,
        is_radian: bool = False,
    ) -> tuple[int, list[float] | float]:
        code_j, servo_angles = self._arm.get_servo_angle(is_radian=True)
        if int(code_j) != 0:
            return int(code_j), [] if servo_id is None else 0.0
        vals = [float(q) if is_radian else float(math.degrees(q)) for q in servo_angles[:6]]
        if servo_id is not None:
            idx = int(servo_id) - 1
            if not (0 <= idx < 6):
                return XARM_ERR_INVALID_PARAM, 0.0
            return XARM_OK, vals[idx]
        return XARM_OK, vals

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
        del mvtime
        self._require_writable("set_servo_angle")
        if angle is None:
            return XARM_ERR_INVALID_PARAM
        spd_rad = 0.35
        if speed is not None and float(speed) > 0:
            spd_rad = float(speed) if is_radian else math.radians(float(speed))
        acc_rad = 1.0 if mvacc is None else (float(mvacc) if is_radian else math.radians(float(mvacc)))

        if isinstance(angle, (int, float)):
            ang_in: list[float] | float = float(angle) if is_radian else math.radians(float(angle))
        else:
            arr = np.asarray(angle, dtype=float).reshape(-1)[:6]
            ang_in = (arr if is_radian else np.deg2rad(arr)).tolist()

        self._prepare_motion()
        code = int(
            self._arm.set_servo_angle(
                servo_id=servo_id,
                angle=ang_in,
                speed=spd_rad,
                mvacc=acc_rad,
                relative=relative,
                is_radian=True,
                wait=wait,
            )
        )
        if code != 0:
            raise XArmSafetyError(f"set_servo_angle failed with code={code}")
        return XARM_OK

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
        del radius, mvtime
        self._require_writable("set_position")

        _, cur_tcp = self.get_position(is_radian=True)
        target_mm_rad = np.array(cur_tcp, dtype=float)

        for idx, val in enumerate((x, y, z)):
            if val is not None:
                target_mm_rad[idx] = (target_mm_rad[idx] + float(val)) if relative else float(val)

        for idx, val in enumerate((roll, pitch, yaw)):
            if val is not None:
                v_rad = float(val) if is_radian else math.radians(float(val))
                target_mm_rad[3 + idx] = (target_mm_rad[3 + idx] + v_rad) if relative else v_rad

        target_quat = rpy_to_quat_wxyz(
            float(target_mm_rad[3]), float(target_mm_rad[4]), float(target_mm_rad[5])
        )
        target_pose = Pose.from_xyz(
            float(target_mm_rad[0] / 1000.0),
            float(target_mm_rad[1] / 1000.0),
            float(target_mm_rad[2] / 1000.0),
            target_quat,
        )

        spd_m_s = 0.10 if (speed is None or float(speed) <= 0) else float(speed) / 1000.0
        spd_m_s = min(spd_m_s, 0.15)
        try:
            self.validator.validate_pose(target_pose, label="xarm_set_position")
            vel_m_s = self.validator.validate_speed(spd_m_s)
        except WorkspaceViolationError:
            return XARM_ERR_OUT_OF_RANGE

        flange_pos_m, flange_quat = tcp_mm_rpy_to_flange_m_quat(
            target_mm_rad, self._tcp_offset_mm_rad
        )
        flange_xyz_mm = flange_pos_m * 1000.0
        f_roll, f_pitch, f_yaw = matrix_to_rpy(
            rpy_to_matrix(
                *flange_m_quat_to_tcp_mm_rpy(flange_pos_m, flange_quat, None)[3:6]
            )
        )

        self._prepare_motion()
        code = int(
            self._arm.set_position(
                x=float(flange_xyz_mm[0]),
                y=float(flange_xyz_mm[1]),
                z=float(flange_xyz_mm[2]),
                roll=float(f_roll),
                pitch=float(f_pitch),
                yaw=float(f_yaw),
                speed=float(vel_m_s * 1000.0),
                mvacc=500.0 if mvacc is None else float(mvacc),
                is_radian=True,
                wait=wait,
            )
        )
        if code != 0:
            raise XArmSafetyError(f"set_position failed with code={code}")
        return XARM_OK

    def reset(
        self,
        speed: float | None = None,
        mvacc: float | None = None,
        mvtime: float | None = None,
        is_radian: bool = False,
        wait: bool = True,
    ) -> int:
        del mvtime, is_radian
        return self.set_servo_angle(
            angle=list(self.home_joints_rad),
            speed=0.35 if speed is None else speed,
            mvacc=1.0 if mvacc is None else mvacc,
            is_radian=True,
            wait=wait,
        )

    # ------------------------------------------------------------------
    # Standard XArmAPI Gripper & Gripper G2 Methods
    # ------------------------------------------------------------------
    def set_gripper_enable(self, enable: bool = True) -> int:
        self._require_writable("set_gripper_enable")
        return int(self._arm.set_gripper_enable(bool(enable)))

    def set_gripper_mode(self, mode: int = 0) -> int:
        self._require_writable("set_gripper_mode")
        return int(self._arm.set_gripper_mode(int(mode)))

    def set_gripper_speed(self, speed: float) -> int:
        self._require_writable("set_gripper_speed")
        return int(self._arm.set_gripper_speed(float(speed)))

    def get_gripper_position(self) -> tuple[int, float]:
        try:
            code_g, raw_g = self._arm.get_gripper_position()
            if int(code_g) == 0 and raw_g is not None:
                return XARM_OK, float(raw_g)
        except (ValueError, TypeError, RuntimeError):
            pass
        return XARM_OK, float(sdk_position_counts(GRIPPER_G2_OPEN_GAP_M * 1000.0))

    def set_gripper_position(
        self,
        pos: float,
        wait: bool = True,
        speed: float | None = None,
    ) -> int:
        self._require_writable("set_gripper_position")
        self._arm.set_gripper_enable(True)
        self._arm.set_gripper_mode(0)
        spd = 1200 if speed is None else int(speed)
        code = int(self._arm.set_gripper_position(float(pos), wait=wait, speed=spd))
        if code != 0:
            raise XArmSafetyError(f"set_gripper_position failed with code={code}")
        return XARM_OK

    def set_gripper_g2_position(
        self,
        pos: int,
        speed: int = 100,
        force: int = 50,
        wait: bool = True,
    ) -> int:
        self._require_writable("set_gripper_g2_position")
        g2_fn = getattr(self._arm, "set_gripper_g2_position", None)
        if callable(g2_fn):
            code = int(g2_fn(int(pos), speed=int(speed), force=int(force), wait=wait))
            if code != 0:
                raise XArmSafetyError(f"set_gripper_g2_position failed with code={code}")
            return XARM_OK
        counts = sdk_position_counts(max(0.0, min(84.0, float(pos))))
        return self.set_gripper_position(float(counts), wait=wait, speed=1200)

    def get_robot_state(self) -> RobotState:
        code_j, servo_angles = self.get_servo_angle(is_radian=True)
        if code_j != 0 or not isinstance(servo_angles, list):
            raise XArmSafetyError(f"get_servo_angle failed with code={code_j}")
        joints = tuple(float(q) for q in servo_angles[:6])

        flange_pose = self._get_flange_pose_m_quat()
        _, tcp_mm_rad = self.get_position(is_radian=True)
        tcp_quat = rpy_to_quat_wxyz(float(tcp_mm_rad[3]), float(tcp_mm_rad[4]), float(tcp_mm_rad[5]))
        tcp_pose = Pose.from_xyz(
            float(tcp_mm_rad[0] / 1000.0),
            float(tcp_mm_rad[1] / 1000.0),
            float(tcp_mm_rad[2] / 1000.0),
            tcp_quat,
        )

        _, raw_counts = self.get_gripper_position()
        gap_m = max(0.0, min(GRIPPER_G2_OPEN_GAP_M, sdk_position_mm(float(raw_counts)) / 1000.0))

        return RobotState(
            joint_positions_rad=joints,
            tcp_pose=tcp_pose,
            flange_pose=flange_pose,
            gripper_gap_m=gap_m,
            is_grasping=gap_m < 0.05,
            contact_forces_n=(0.0, 0.0),
        )
