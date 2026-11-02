"""MuJoCo simulation backend implementing XArmAPI-compatible RobotBackend for xArm 6 + Gripper G2."""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from pathlib import Path

import cv2
import mujoco
import numpy as np

from core.backend import RobotBackend
from core.safety import (
    XARM_ERR_COLLISION,
    XARM_ERR_IK_FAILED,
    XARM_ERR_INVALID_PARAM,
    XARM_ERR_NOT_READY,
    XARM_ERR_OUT_OF_RANGE,
    XARM_OK,
    CollisionChecker,
    WorkspaceValidator,
    WorkspaceViolationError,
)
from core.transforms import (
    apply_calibration_to_model,
    flange_m_quat_to_tcp_mm_rpy,
    load_kinematics_calibration,
    rpy_to_quat_wxyz,
)
from core.types import GRIP_DOWN_QUAT_WXYZ, Pose, RobotState, WorkspaceLimits

from .ik import (
    GRIPPER_G2_OPEN_GAP_M,
    gripper_g2_gap_m_to_sim_drive,
    gripper_g2_sim_drive_to_gap_m,
    quintic_interpolate,
    sdk_position_counts,
    sdk_position_mm,
    solve_ik,
    solve_ik_trajectory,
)
from .viewer import PassiveViewer, VideoRecorder

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_SCENE_XML = PROJECT_ROOT / "assets" / "models" / "xarm6_with_gripper" / "scene_pick_place.xml"
DEFAULT_KINEMATICS_YAML = PROJECT_ROOT / "configs" / "xarm6_kinematics_L56A01.yaml"


class MuJoCoBackend(RobotBackend):
    """MuJoCo physics simulation backend with 100% XArmAPI-compatible control methods."""

    def __init__(
        self,
        scene_xml: str | Path = DEFAULT_SCENE_XML,
        limits: WorkspaceLimits | None = None,
        cart_acc_m_s2: float = 0.80,
        *,
        calibration_path: str | Path | None = None,
        viewer: bool = False,
        viewer_hz: float = 30.0,
        record_video_path: str | Path | None = None,
    ) -> None:
        self.scene_xml = Path(scene_xml)
        if not self.scene_xml.is_file():
            raise FileNotFoundError(f"MuJoCo scene XML not found: {self.scene_xml}")

        self.validator = WorkspaceValidator(limits)
        self.cart_acc_m_s2 = float(cart_acc_m_s2)

        self.model = mujoco.MjModel.from_xml_path(str(self.scene_xml))
        self.data = mujoco.MjData(self.model)
        self.eval_data = mujoco.MjData(self.model)
        self.collision_checker = CollisionChecker(self.scene_xml)

        self.serial_number = "XI130505L56A01"
        if calibration_path is not None:
            calib = load_kinematics_calibration(calibration_path)
            self.serial_number = calib.serial_number
            apply_calibration_to_model(self.model, calib, self.data)
            apply_calibration_to_model(self.collision_checker.model, calib, self.collision_checker.data)
            mujoco.mj_setConst(self.model, self.eval_data)
            mujoco.mj_forward(self.model, self.eval_data)

        self._tcp_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
        if self._tcp_site_id < 0:
            raise ValueError("Model is missing 'attachment_site'")
        self._flange_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "link6")
        self._n_arm = 6
        self._gripper_act_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "gripper")
        drv_jnt = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "left_driver_joint")
        self._drv_qpos_adr = int(self.model.jnt_qposadr[drv_jnt]) if drv_jnt >= 0 else 6

        # Default TCP offset: 173.5 mm along flange +Z (Gripper G2 fingertip center)
        self._tcp_offset_mm_rad = np.array([0.0, 0.0, 173.5, 0.0, 0.0, 0.0], dtype=float)
        self._motion_enabled = True
        self._mode = 0
        self._state = 2  # 1=in motion, 2=ready/standby, 3=paused, 4=stopped
        self._error_code = 0
        self._warn_code = 0
        self._gripper_enabled = True
        self._gripper_mode = 0
        self._gripper_speed = 1500.0

        # Allow full force range on Gripper G2 actuator for firm grasp
        if self._gripper_act_id >= 0:
            self.model.actuator_ctrllimited[self._gripper_act_id] = 0
            self.model.actuator_forcerange[self._gripper_act_id] = (-12.0, 12.0)

        self._viewer: PassiveViewer | None = None
        self._viewer_interval = 1.0 / max(1.0, float(viewer_hz))
        self._last_viewer_sync = 0.0
        self._wall_start = 0.0

        self._recorder: VideoRecorder | None = None
        self._record_interval = 1.0 / 30.0
        self._next_record_sim_time = 0.0

        self.reset_scene()

        if viewer:
            self._viewer = PassiveViewer(self.model, self.data)
            self._wall_start = time.monotonic() - float(self.data.time)
            self._last_viewer_sync = time.monotonic()
            self._viewer.sync()

        if record_video_path is not None:
            self._recorder = VideoRecorder(self.model, record_video_path, fps=30)
            self._recorder.capture(self.data)
            self._next_record_sim_time = float(self.data.time) + self._record_interval

    # ------------------------------------------------------------------
    # Simulation Lifecycle & Physics Helpers
    # ------------------------------------------------------------------
    def reset_scene(self) -> None:
        """Reset robot to home keyframe while preserving free-joint object qpos0."""
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        for j in range(self.model.njnt):
            if int(self.model.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_FREE):
                adr = int(self.model.jnt_qposadr[j])
                self.data.qpos[adr : adr + 7] = self.model.qpos0[adr : adr + 7]
        self.data.ctrl[: self._n_arm] = self.data.qpos[: self._n_arm]
        if self._gripper_act_id >= 0:
            self.data.ctrl[self._gripper_act_id] = gripper_g2_gap_m_to_sim_drive(GRIPPER_G2_OPEN_GAP_M)
        mujoco.mj_forward(self.model, self.data)
        self._error_code = 0
        self._warn_code = 0
        self._state = 2
        self._wall_start = time.monotonic() - float(self.data.time)

    def _step_physics(self) -> None:
        self.data.qfrc_applied[: self._n_arm] = self.data.qfrc_bias[: self._n_arm]
        mujoco.mj_step(self.model, self.data)
        self.data.qfrc_applied[:] = 0.0

        if self._recorder is not None and float(self.data.time) >= self._next_record_sim_time:
            self._recorder.capture(self.data)
            self._next_record_sim_time += self._record_interval

        if self._viewer is not None and self._viewer.is_running():
            wall_elapsed = time.monotonic() - self._wall_start
            sim_ahead = float(self.data.time) - wall_elapsed
            if sim_ahead > 0:
                time.sleep(sim_ahead)
            now = time.monotonic()
            if now - self._last_viewer_sync >= self._viewer_interval:
                self._viewer.sync()
                self._last_viewer_sync = now

    def wait_viewer(self) -> None:
        """Keep the interactive 3D viewer open until the user closes the window."""
        if self._viewer is None:
            return
        print("Viewer is open. Press Esc or close the window to exit...")
        while self._viewer.is_running():
            self._viewer.sync()
            time.sleep(self._viewer_interval)

    def sync_viewer(self) -> None:
        """Refresh the 3D viewer if open."""
        if self._viewer is not None and self._viewer.is_running():
            self._viewer.sync()

    def close(self) -> Path | None:
        """Close viewer and finalize video recording if enabled."""
        saved_video: Path | None = None
        if self._recorder is not None:
            saved_video = self._recorder.save()
            self._recorder = None
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
        return saved_video

    def settle(self, duration_s: float = 0.3) -> None:
        n_steps = max(1, round(duration_s / self.model.opt.timestep))
        for _ in range(n_steps):
            if self._viewer is not None and not self._viewer.is_running():
                break
            self._step_physics()

    def render_snapshot(
        self,
        output_path: str | Path,
        *,
        camera: str = "overview",
        width: int = 640,
        height: int = 480,
    ) -> Path:
        """Render an offscreen RGB snapshot of the simulation and save as PNG."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        mujoco.mj_forward(self.model, self.data)
        renderer = mujoco.Renderer(self.model, height=height, width=width)
        try:
            cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
            if cam_id >= 0:
                renderer.update_scene(self.data, camera=camera)
            else:
                mj_cam = mujoco.MjvCamera()
                mujoco.mjv_defaultFreeCamera(self.model, mj_cam)
                mj_cam.lookat[:] = (0.30, 0.10, 0.12)
                mj_cam.distance = 1.05
                mj_cam.azimuth = 135.0
                mj_cam.elevation = -28.0
                renderer.update_scene(self.data, camera=mj_cam)
            rgb = renderer.render()
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(out), bgr)
        finally:
            renderer.close()
        return out

    # ------------------------------------------------------------------
    # Internal Pose & Contact Force Queries
    # ------------------------------------------------------------------
    def get_tcp_pose(self) -> Pose:
        flange = self.get_flange_pose()
        tcp_mm_rad = flange_m_quat_to_tcp_mm_rpy(
            flange.position, flange.orientation_wxyz, self._tcp_offset_mm_rad
        )
        quat = rpy_to_quat_wxyz(float(tcp_mm_rad[3]), float(tcp_mm_rad[4]), float(tcp_mm_rad[5]))
        return Pose.from_xyz(
            float(tcp_mm_rad[0] / 1000.0),
            float(tcp_mm_rad[1] / 1000.0),
            float(tcp_mm_rad[2] / 1000.0),
            quat,
        )

    def get_flange_pose(self) -> Pose:
        pos = self.data.body(self._flange_body_id).xpos.copy()
        quat = self.data.body(self._flange_body_id).xquat.copy()
        return Pose.from_xyz(float(pos[0]), float(pos[1]), float(pos[2]), quat)

    def get_gripper_gap_m(self) -> float:
        drv_q = float(self.data.qpos[self._drv_qpos_adr])
        return gripper_g2_sim_drive_to_gap_m(drv_q)

    def bilateral_finger_contact(self, object_body: str = "cube", max_dist: float = 1e-4) -> bool:
        """Check whether both left_finger and right_finger are in contact with object_body."""
        left_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "left_finger")
        right_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "right_finger")
        obj_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, object_body)
        if min(left_id, right_id, obj_id) < 0:
            return False

        left_touch = False
        right_touch = False
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            if c.dist > max_dist:
                continue
            pair = {int(self.model.geom_bodyid[c.geom1]), int(self.model.geom_bodyid[c.geom2])}
            if pair == {left_id, obj_id}:
                left_touch = True
            elif pair == {right_id, obj_id}:
                right_touch = True
        return left_touch and right_touch

    def get_gripper_contact_forces_n(self, object_body: str = "cube") -> tuple[float, float]:
        """Return net contact force magnitudes (N) on (left_finger, right_finger) for object_body."""
        left_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "left_finger")
        right_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "right_finger")
        obj_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, object_body)
        if min(left_id, right_id, obj_id) < 0:
            return (0.0, 0.0)

        left_net = np.zeros(3, dtype=float)
        right_net = np.zeros(3, dtype=float)
        wrench = np.zeros(6, dtype=float)

        for i in range(self.data.ncon):
            c = self.data.contact[i]
            b1 = int(self.model.geom_bodyid[c.geom[0]])
            b2 = int(self.model.geom_bodyid[c.geom[1]])
            if obj_id not in (b1, b2):
                continue
            other = b2 if b1 == obj_id else b1
            if other not in (left_id, right_id):
                continue
            mujoco.mj_contactForce(self.model, self.data, i, wrench)
            frame = np.asarray(c.frame, dtype=float).reshape(3, 3)
            force_world = frame.T @ wrench[:3]
            applied_to_finger = force_world if other == b2 else -force_world
            if other == left_id:
                left_net += applied_to_finger
            else:
                right_net += applied_to_finger

        return float(np.linalg.norm(left_net)), float(np.linalg.norm(right_net))

    def get_robot_state(self) -> RobotState:
        joints = tuple(float(q) for q in self.data.qpos[: self._n_arm])
        return RobotState(
            joint_positions_rad=joints,
            tcp_pose=self.get_tcp_pose(),
            flange_pose=self.get_flange_pose(),
            gripper_gap_m=self.get_gripper_gap_m(),
            is_grasping=self.bilateral_finger_contact("cube"),
            contact_forces_n=self.get_gripper_contact_forces_n("cube"),
        )

    # ------------------------------------------------------------------
    # Standard XArmAPI System & State Methods
    # ------------------------------------------------------------------
    def motion_enable(self, enable: bool = True, servo_id: int | None = None) -> int:
        del servo_id
        self._motion_enabled = bool(enable)
        return XARM_OK

    def set_mode(self, mode: int = 0) -> int:
        self._mode = int(mode)
        return XARM_OK

    def set_state(self, state: int = 0) -> int:
        s = int(state)
        self._state = 2 if s == 0 else s
        return XARM_OK

    def clean_error(self) -> int:
        self._error_code = 0
        if self._state == 4:
            self._state = 2
        return XARM_OK

    def clean_warn(self) -> int:
        self._warn_code = 0
        return XARM_OK

    def emergency_stop(self) -> None:
        self._state = 4
        self.data.ctrl[: self._n_arm] = self.data.qpos[: self._n_arm]

    def get_state(self) -> tuple[int, int]:
        return XARM_OK, int(self._state)

    def get_err_warn_code(self) -> tuple[int, list[int]]:
        return XARM_OK, [int(self._error_code), int(self._warn_code)]

    def get_robot_sn(self) -> tuple[int, str]:
        return XARM_OK, self.serial_number

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
        self.model.site_pos[self._tcp_site_id] = arr[:3] / 1000.0
        self.model.site_quat[self._tcp_site_id] = rpy_to_quat_wxyz(
            float(rpy_rad[0]), float(rpy_rad[1]), float(rpy_rad[2])
        )
        mujoco.mj_setConst(self.model, self.data)
        mujoco.mj_setConst(self.model, self.eval_data)
        mujoco.mj_forward(self.model, self.data)
        return XARM_OK

    def get_position(self, is_radian: bool = False) -> tuple[int, list[float]]:
        flange = self.get_flange_pose()
        tcp_mm_rad = flange_m_quat_to_tcp_mm_rpy(
            flange.position, flange.orientation_wxyz, self._tcp_offset_mm_rad
        )
        if is_radian:
            return XARM_OK, [float(v) for v in tcp_mm_rad]
        out = [
            float(tcp_mm_rad[0]),
            float(tcp_mm_rad[1]),
            float(tcp_mm_rad[2]),
            float(math.degrees(tcp_mm_rad[3])),
            float(math.degrees(tcp_mm_rad[4])),
            float(math.degrees(tcp_mm_rad[5])),
        ]
        return XARM_OK, out

    def get_servo_angle(
        self,
        servo_id: int | None = None,
        is_radian: bool = False,
    ) -> tuple[int, list[float] | float]:
        q_rad = self.data.qpos[: self._n_arm].copy()
        vals = [float(q) if is_radian else float(math.degrees(q)) for q in q_rad]
        if servo_id is not None:
            idx = int(servo_id) - 1
            if not (0 <= idx < self._n_arm):
                return XARM_ERR_INVALID_PARAM, 0.0
            return XARM_OK, vals[idx]
        return XARM_OK, vals

    def move_joints(self, q_goal: np.ndarray, duration_s: float = 1.0, wait: bool = True) -> None:
        q_target = np.asarray(q_goal, dtype=float).reshape(self._n_arm)
        q0 = self.data.ctrl[: self._n_arm].copy()
        dt = float(self.model.opt.timestep)
        n_steps = max(1, int(np.ceil(duration_s / dt)))
        times = np.minimum(np.arange(1, n_steps + 1) * dt, duration_s)
        traj = quintic_interpolate(times, q0, q_target, duration_s)
        self._state = 1
        for q in traj:
            if self._state == 4 or (self._viewer is not None and not self._viewer.is_running()):
                return
            self.data.ctrl[: self._n_arm] = q
            self._step_physics()
        if wait:
            self.settle(0.25)
        if self._state == 1:
            self._state = 2

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
        del mvacc, mvtime
        if not self._motion_enabled or self._state == 4 or self._error_code != 0:
            return XARM_ERR_NOT_READY
        if angle is None:
            return XARM_ERR_INVALID_PARAM

        q_cur = self.data.qpos[: self._n_arm].copy()
        q_goal = q_cur.copy()

        if servo_id is not None and servo_id != 8:
            idx = int(servo_id) - 1
            if not (0 <= idx < self._n_arm):
                return XARM_ERR_INVALID_PARAM
            if isinstance(angle, Sequence):
                return XARM_ERR_INVALID_PARAM
            val_rad = float(angle) if is_radian else math.radians(float(angle))
            q_goal[idx] = (q_cur[idx] + val_rad) if relative else val_rad
        else:
            if isinstance(angle, (int, float)):
                return XARM_ERR_INVALID_PARAM
            arr = np.asarray(angle, dtype=float).reshape(-1)[: self._n_arm]
            if arr.shape != (self._n_arm,):
                return XARM_ERR_INVALID_PARAM
            arr_rad = arr if is_radian else np.deg2rad(arr)
            q_goal = (q_cur + arr_rad) if relative else arr_rad

        if not self.collision_checker.within_joint_limits(q_goal):
            self._error_code = XARM_ERR_OUT_OF_RANGE
            return XARM_ERR_OUT_OF_RANGE
        if self.collision_checker.in_collision(q_goal):
            self._error_code = XARM_ERR_COLLISION
            return XARM_ERR_COLLISION

        spd_rad_s = 1.0
        if speed is not None and float(speed) > 0:
            spd_rad_s = float(speed) if is_radian else math.radians(float(speed))
        max_disp = float(np.max(np.abs(q_goal - q_cur)))
        duration = max(0.25, max_disp / max(0.05, spd_rad_s) + 0.15)
        self.move_joints(q_goal, duration_s=duration, wait=wait)
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
        del radius, mvacc, mvtime
        if not self._motion_enabled or self._state == 4 or self._error_code != 0:
            return XARM_ERR_NOT_READY

        _, cur_tcp = self.get_position(is_radian=True)
        target_mm_rad = np.array(cur_tcp, dtype=float)

        coords_mm = (x, y, z)
        for idx, val in enumerate(coords_mm):
            if val is not None:
                target_mm_rad[idx] = (target_mm_rad[idx] + float(val)) if relative else float(val)

        angles_in = (roll, pitch, yaw)
        for idx, val in enumerate(angles_in):
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

        # Speed in XArmAPI is mm/s; convert to m/s (default 150 mm/s = 0.15 m/s)
        spd_m_s = 0.15 if (speed is None or float(speed) <= 0) else float(speed) / 1000.0
        spd_m_s = min(spd_m_s, self.validator.limits.max_cart_vel_m_s)

        try:
            self.validator.validate_pose(target_pose, label="set_position")
            vel = self.validator.validate_speed(spd_m_s)
        except WorkspaceViolationError:
            self._error_code = XARM_ERR_OUT_OF_RANGE
            return XARM_ERR_OUT_OF_RANGE

        cur_pose = self.get_tcp_pose()
        pos0 = cur_pose.pos_np
        quat0 = cur_pose.quat_np
        pos_goal = target_pose.pos_np
        quat_goal = target_pose.quat_np

        dist = float(np.linalg.norm(pos_goal - pos0))
        duration = max(0.25, dist / vel + vel / self.cart_acc_m_s2 + 0.1)
        steps = max(20, min(60, int(np.ceil(dist / 0.005)) + 15))

        s = np.linspace(0.0, 1.0, steps + 1)[1:]
        waypoints = pos0 + s[:, None] * (pos_goal - pos0)

        q0_conj = np.array([quat0[0], -quat0[1], -quat0[2], -quat0[3]], dtype=float)
        q_delta_local = np.empty(4, dtype=float)
        mujoco.mju_mulQuat(q_delta_local, q0_conj, quat_goal)
        rotvec_local = np.empty(3, dtype=float)
        mujoco.mju_quat2Vel(rotvec_local, q_delta_local, 1.0)
        quat_traj = np.empty((steps, 4), dtype=float)
        for i in range(steps):
            quat_traj[i] = quat0
            mujoco.mju_quatIntegrate(quat_traj[i], rotvec_local, s[i])

        q_start = self.data.qpos.copy()
        q_traj, ok = solve_ik_trajectory(
            self.model,
            self.eval_data,
            "attachment_site",
            waypoints,
            quat_traj,
            q_start,
        )
        if not ok:
            self._error_code = XARM_ERR_IK_FAILED
            return XARM_ERR_IK_FAILED

        # Check joint limits and self-collision on the final configuration
        q_final = q_traj[-1][: self._n_arm]
        if not self.collision_checker.within_joint_limits(q_final):
            self._error_code = XARM_ERR_OUT_OF_RANGE
            return XARM_ERR_OUT_OF_RANGE
        if self.collision_checker.in_collision(q_final):
            self._error_code = XARM_ERR_COLLISION
            return XARM_ERR_COLLISION

        seg_dt = max(duration / steps, 4 * float(self.model.opt.timestep))
        for q in q_traj:
            self.move_joints(q[: self._n_arm], duration_s=seg_dt, wait=False)
        if wait:
            self.settle(0.25)
        return XARM_OK

    def reset(
        self,
        speed: float | None = None,
        mvacc: float | None = None,
        mvtime: float | None = None,
        is_radian: bool = False,
        wait: bool = True,
    ) -> int:
        del mvacc, mvtime
        home_q = self.model.key("home").qpos[: self._n_arm].copy()
        return self.set_servo_angle(angle=home_q.tolist(), speed=speed, is_radian=True, wait=wait)

    # ------------------------------------------------------------------
    # Standard XArmAPI Gripper & Gripper G2 Methods
    # ------------------------------------------------------------------
    def set_gripper_enable(self, enable: bool = True) -> int:
        self._gripper_enabled = bool(enable)
        return XARM_OK

    def set_gripper_mode(self, mode: int = 0) -> int:
        self._gripper_mode = int(mode)
        return XARM_OK

    def set_gripper_speed(self, speed: float) -> int:
        self._gripper_speed = max(1.0, float(speed))
        return XARM_OK

    def get_gripper_position(self) -> tuple[int, float]:
        gap_mm = self.get_gripper_gap_m() * 1000.0
        return XARM_OK, float(sdk_position_counts(gap_mm))

    def set_gripper_gap(self, gap_m: float, settle_s: float = 0.4) -> None:
        if self._gripper_act_id < 0:
            return
        drive_rad = gripper_g2_gap_m_to_sim_drive(gap_m)
        self.data.ctrl[self._gripper_act_id] = drive_rad
        if settle_s > 0:
            self.settle(settle_s)

    def set_gripper_position(
        self,
        pos: float,
        wait: bool = True,
        speed: float | None = None,
    ) -> int:
        if not self._gripper_enabled:
            return XARM_ERR_NOT_READY
        if speed is not None:
            self.set_gripper_speed(speed)
        counts = float(pos)
        if not math.isfinite(counts) or counts < -10.0 or counts > 860.0:
            return XARM_ERR_INVALID_PARAM
        gap_mm = max(0.0, min(84.0, sdk_position_mm(max(0.0, min(840.0, counts)))))
        gap_m = gap_mm / 1000.0
        cur_gap_m = self.get_gripper_gap_m()
        closing = gap_m < cur_gap_m - 0.005

        self.set_gripper_gap(gap_m, settle_s=0.8 if wait else 0.0)
        if wait and closing and gap_m <= 0.026:
            # If grasping near an object in 3D, step progressively to establish bilateral pad contact
            cube_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cube")
            if cube_id >= 0:
                cube_pos = self.data.body(cube_id).xpos
                tcp_pos = self.get_tcp_pose().pos_np
                if float(np.linalg.norm(cube_pos - tcp_pos)) < 0.035:
                    step_gap = gap_m
                    while not self.bilateral_finger_contact("cube") and step_gap > 0.0:
                        step_gap = max(0.0, step_gap - 0.002)
                        self.set_gripper_gap(step_gap, settle_s=0.2)
        return XARM_OK

    def set_gripper_g2_position(
        self,
        pos: int,
        speed: int = 100,
        force: int = 50,
        wait: bool = True,
    ) -> int:
        del speed, force
        if not (0 <= int(pos) <= 86):
            return XARM_ERR_INVALID_PARAM
        counts = sdk_position_counts(float(pos))
        return self.set_gripper_position(float(counts), wait=wait)

    def open_gripper(self) -> None:
        self.set_gripper_g2_position(pos=84, speed=150, force=50, wait=True)

    def close_gripper(self, target_gap_m: float = 0.022, target_body: str | None = "cube") -> float:
        gap = float(target_gap_m)
        self.set_gripper_gap(gap, settle_s=1.2)
        body_name = target_body or "cube"
        while not self.bilateral_finger_contact(body_name) and gap > 0.0:
            gap = max(0.0, gap - 0.002)
            self.set_gripper_gap(gap, settle_s=0.25)
        if not self.bilateral_finger_contact(body_name):
            raise RuntimeError(f"Failed to achieve bilateral finger contact on {body_name!r}")
        return gap

    # ------------------------------------------------------------------
    # Scene Manipulation Helpers (Simulation Only)
    # ------------------------------------------------------------------
    def measure_flange_to_tcp_offset_z(self, ref_xy: tuple[float, float] = (0.30, 0.0), lift_z_m: float = 0.30) -> float:
        """Compute vertical offset (tcp_z - flange_z) when gripper is pointing straight down."""
        q_seed, ok = solve_ik(
            self.model,
            self.eval_data,
            "attachment_site",
            np.array([ref_xy[0], ref_xy[1], lift_z_m + 0.15]),
            GRIP_DOWN_QUAT_WXYZ,
        )
        if not ok:
            raise RuntimeError("IK failed while solving seed pose")
        q, ok = solve_ik(
            self.model,
            self.eval_data,
            "attachment_site",
            np.array([ref_xy[0], ref_xy[1], lift_z_m]),
            GRIP_DOWN_QUAT_WXYZ,
            q_init=q_seed,
        )
        if not ok:
            raise RuntimeError("IK failed while measuring flange->TCP offset")
        self.eval_data.qpos[:] = q
        mujoco.mj_forward(self.model, self.eval_data)
        tcp_z = float(self.eval_data.site_xpos[self._tcp_site_id][2])
        flange_z = float(self.eval_data.body(self._flange_body_id).xpos[2])
        return tcp_z - flange_z

    def set_cube_xy(self, x: float, y: float) -> None:
        joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "cube")
        if joint_id < 0:
            raise ValueError("Scene has no 'cube' freejoint")
        adr = int(self.model.jnt_qposadr[joint_id])
        authored_z = float(self.model.qpos0[adr + 2])
        self.data.qpos[adr : adr + 3] = (float(x), float(y), authored_z)
        self.data.qpos[adr + 3 : adr + 7] = (1.0, 0.0, 0.0, 0.0)
        dof = int(self.model.jnt_dofadr[joint_id])
        self.data.qvel[dof : dof + 6] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self.sync_viewer()

    def set_target_xy(self, x: float, y: float) -> None:
        geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "target_marker")
        if geom_id < 0:
            raise ValueError("Scene has no 'target_marker' geom")
        authored_z = float(self.model.geom_pos[geom_id][2])
        self.model.geom_pos[geom_id] = (float(x), float(y), authored_z)
        mujoco.mj_forward(self.model, self.data)
        self.sync_viewer()
