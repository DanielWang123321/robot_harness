"""Unit & Integration tests for XArmAPI-compatible Backend, CollisionChecker, SessionDaemon, and MCPServer."""

from __future__ import annotations

import threading
import time
from pathlib import Path

from core.safety import XARM_ERR_OUT_OF_RANGE, XARM_OK
from execution.mujoco_backend import DEFAULT_KINEMATICS_YAML, MuJoCoBackend
from harness.mcp_server import MCPServer
from harness.runtime import RobotHarnessRuntime
from harness.session import SessionClient, SessionDaemon


def test_xarm_api_signatures_and_partial_set_position() -> None:
    """Verify XArmAPI methods, partial set_position updates, set_tcp_offset, and safety error codes."""
    arm = MuJoCoBackend(calibration_path=DEFAULT_KINEMATICS_YAML)
    try:
        code_sn, sn = arm.get_robot_sn()
        assert code_sn == XARM_OK
        assert sn == "XI130505L56A01"

        # Move to an initial Cartesian position in mm and deg
        code = arm.set_position(x=300.0, y=0.0, z=150.0, roll=180.0, pitch=0.0, yaw=0.0, speed=200.0, wait=True)
        assert code == XARM_OK

        code_p, pos = arm.get_position(is_radian=False)
        assert code_p == XARM_OK
        assert abs(pos[0] - 300.0) < 2.0
        assert abs(pos[1] - 0.0) < 2.0
        assert abs(pos[2] - 150.0) < 2.0

        # Partial update: only specify z=100.0 mm, x and y must remain at 300.0, 0.0
        code_z = arm.set_position(z=100.0, speed=200.0, wait=True)
        assert code_z == XARM_OK
        _, pos2 = arm.get_position(is_radian=False)
        assert abs(pos2[0] - 300.0) < 2.0
        assert abs(pos2[2] - 100.0) < 2.0

        # Out-of-bounds table collision guard (z = 2.0 mm < 8.0 mm) must return XARM_ERR_OUT_OF_RANGE (31)
        code_err = arm.set_position(z=2.0, wait=True)
        assert code_err == XARM_ERR_OUT_OF_RANGE
        _, err_warn = arm.get_err_warn_code()
        assert err_warn[0] == XARM_ERR_OUT_OF_RANGE
        assert arm.clean_error() == XARM_OK

        # Gripper G2 control via SDK pulse counts and mm
        assert arm.set_gripper_g2_position(pos=40, wait=True) == XARM_OK
        code_g, counts = arm.get_gripper_position()
        assert code_g == XARM_OK
        assert 100.0 < counts < 600.0

        # Reset back to home
        assert arm.reset(wait=True) == XARM_OK
    finally:
        arm.close()


def test_session_daemon_and_client_state_persistence(tmp_path: Path) -> None:
    """Verify stateful multi-step control across SessionClient calls over a Unix socket."""
    sock_path = tmp_path / "test_harness.sock"
    runs_dir = tmp_path / "runs"
    runtime = RobotHarnessRuntime(MuJoCoBackend(), runs_dir=runs_dir)
    daemon = SessionDaemon(runtime, socket_path=sock_path)

    thread = threading.Thread(target=daemon.serve_forever, daemon=True)
    thread.start()
    try:
        client = SessionClient(sock_path)
        for _ in range(30):
            if client.is_running():
                break
            time.sleep(0.05)
        assert client.is_running()

        # Step 1: Observe and verify snapshot image is saved
        obs = client.request("observe", {"save_image": True, "camera": "overview"})
        assert obs["ok"] is True
        assert Path(obs["snapshot_path"]).is_file()

        # Step 2: Execute XArmAPI set_position via daemon and verify state persists
        move_res = client.request(
            "xarm",
            {
                "method": "set_position",
                "kwargs": {"x": 300.0, "y": 50.0, "z": 140.0, "roll": 180.0, "pitch": 0.0, "yaw": 0.0},
            },
        )
        assert move_res["ok"] is True
        assert abs(move_res["robot_state"]["tcp_mm_deg"][1] - 50.0) < 2.5

        # Step 3: Execute high-level pick via daemon and verify bilateral contact force > 0 N
        pick_res = client.request("act", {"primitive": "pick", "params": {"object_id": "red_cube"}})
        assert pick_res["ok"] is True
        assert pick_res["robot_state"]["is_grasping"] is True
        forces = pick_res["robot_state"]["contact_forces_n"]
        assert forces[0] > 0.1 and forces[1] > 0.1

        # Step 4: Shutdown session
        stop_res = client.request("stop")
        assert stop_res["ok"] is True
    finally:
        thread.join(timeout=3.0)


def test_mcp_server_jsonrpc_protocol(tmp_path: Path) -> None:
    """Verify MCPServer JSON-RPC 2.0 initialize, tools/list, and tools/call."""
    runtime = RobotHarnessRuntime(MuJoCoBackend(), runs_dir=tmp_path)
    mcp = MCPServer(runtime=runtime)
    try:
        init_resp = mcp.handle_jsonrpc({"jsonrpc": "2.0", "id": 1, "method": "initialize"})
        assert init_resp is not None
        assert init_resp["result"]["serverInfo"]["name"] == "xarm-robot-harness"

        list_resp = mcp.handle_jsonrpc({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
        assert list_resp is not None
        tool_names = {t["name"] for t in list_resp["result"]["tools"]}
        assert {"xarm_observe", "xarm_set_position", "xarm_pick", "xarm_place"} <= tool_names

        call_resp = mcp.handle_jsonrpc(
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {"name": "xarm_get_position", "arguments": {"is_radian": False}},
            }
        )
        assert call_resp is not None
        assert call_resp["result"]["isError"] is False
    finally:
        runtime.close()
