#!/usr/bin/env python3
"""CLI entrypoint for XArmAPI-Aligned Robot Control Harness & One-Shot Instruction Runner."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from execution.mujoco_backend import MuJoCoBackend
from execution.xarm_backend import XArmBackend
from harness.mcp_server import MCPServer
from harness.runtime import RobotHarnessRuntime
from harness.session import DEFAULT_SOCKET_PATH, SessionClient, SessionDaemon
from orchestrator import TaskOrchestrator
from perception.sim_gt import SimGTPerception
from planning.llm_planner import LLMPlanner
from planning.primitives import PickPlaceHeightParams

SUBCOMMANDS = {"session", "observe", "xarm", "act", "reset", "mcp"}


def _print_json(data: dict[str, Any]) -> int:
    print(json.dumps(data, ensure_ascii=False, indent=2))
    code = data.get("code", 0)
    ok = data.get("ok", code == 0)
    return 0 if (ok and code == 0) else 1


def _run_harness_subcommand(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="main.py", description="Robot Control Harness CLI")
    sub = parser.add_subparsers(dest="subcmd", required=True)

    # session start | status | stop
    p_sess = sub.add_parser("session", help="Manage persistent Robot Harness session daemon")
    p_sess.add_argument("action", choices=["start", "status", "stop"])
    p_sess.add_argument("--backend", choices=["sim", "real"], default="sim")
    p_sess.add_argument("--ip", default="192.168.1.60")
    p_sess.add_argument("--confirm-real", action="store_true")
    p_sess.add_argument("--viewer", action="store_true", help="Open interactive 3D MuJoCo viewer")
    p_sess.add_argument("--save-video", type=Path, default=None, help="Record session to MP4")
    p_sess.add_argument("--socket", type=Path, default=DEFAULT_SOCKET_PATH)

    # observe
    p_obs = sub.add_parser("observe", help="Observe robot state, 3D objects, and save RGB snapshot")
    p_obs.add_argument("--camera", choices=["overview", "eye_in_hand"], default="overview")
    p_obs.add_argument("--no-image", action="store_true")
    p_obs.add_argument("--socket", type=Path, default=DEFAULT_SOCKET_PATH)

    # xarm <method>
    p_xarm = sub.add_parser("xarm", help="Call any standard XArmAPI method directly")
    p_xarm.add_argument("method", help="XArmAPI method name (e.g. get_position, set_position, set_gripper_position)")
    p_xarm.add_argument("--args", default="{}", help="JSON object of keyword arguments")
    p_xarm.add_argument("--x", type=float, default=None)
    p_xarm.add_argument("--y", type=float, default=None)
    p_xarm.add_argument("--z", type=float, default=None)
    p_xarm.add_argument("--roll", type=float, default=None)
    p_xarm.add_argument("--pitch", type=float, default=None)
    p_xarm.add_argument("--yaw", type=float, default=None)
    p_xarm.add_argument("--speed", type=float, default=None)
    p_xarm.add_argument("--pos", type=float, default=None)
    p_xarm.add_argument("--save-image", action="store_true")
    p_xarm.add_argument("--camera", choices=["overview", "eye_in_hand"], default="overview")
    p_xarm.add_argument("--socket", type=Path, default=DEFAULT_SOCKET_PATH)

    # act <primitive>
    p_act = sub.add_parser("act", help="Execute a high-level skill (pick, place, move_home, etc.)")
    p_act.add_argument("primitive", choices=["pick", "place", "move_home", "open_gripper", "close_gripper", "move_tcp"])
    p_act.add_argument("--object-id", default=None)
    p_act.add_argument("--target-id", default=None)
    p_act.add_argument("--params", default="{}", help="Additional JSON parameters")
    p_act.add_argument("--no-image", action="store_true")
    p_act.add_argument("--camera", choices=["overview", "eye_in_hand"], default="overview")
    p_act.add_argument("--socket", type=Path, default=DEFAULT_SOCKET_PATH)

    # reset
    p_rst = sub.add_parser("reset", help="Reset simulation scene props and move robot arm home")
    p_rst.add_argument("--cube-x", type=float, default=0.30)
    p_rst.add_argument("--cube-y", type=float, default=0.00)
    p_rst.add_argument("--target-x", type=float, default=0.30)
    p_rst.add_argument("--target-y", type=float, default=0.30)
    p_rst.add_argument("--no-image", action="store_true")
    p_rst.add_argument("--socket", type=Path, default=DEFAULT_SOCKET_PATH)

    # mcp
    sub.add_parser("mcp", help="Run stdio MCP Server for External Agent integration")

    args = parser.parse_args(argv)

    if args.subcmd == "mcp":
        MCPServer().serve_stdio()
        return 0

    if args.subcmd == "session":
        client = SessionClient(args.socket)
        if args.action == "status":
            if not client.is_running():
                return _print_json({"ok": False, "status": "stopped", "socket": str(args.socket)})
            return _print_json(client.request("status"))
        if args.action == "stop":
            if not client.is_running():
                return _print_json({"ok": True, "status": "already_stopped"})
            return _print_json(client.request("stop"))
        if args.action == "start":
            if args.backend == "real":
                backend = XArmBackend(ip=args.ip, read_only=not args.confirm_real)
            else:
                backend = MuJoCoBackend(viewer=args.viewer, record_video_path=args.save_video)
            runtime = RobotHarnessRuntime(backend)
            daemon = SessionDaemon(runtime, socket_path=args.socket)
            print(f"Robot Harness session started at {args.socket} (backend={args.backend}, viewer={args.viewer})", flush=True)
            daemon.serve_forever()
            return 0

    client = SessionClient(args.socket)
    use_daemon = client.is_running()

    if args.subcmd == "observe":
        req_args = {"save_image": not args.no_image, "camera": args.camera}
        if use_daemon:
            return _print_json(client.request("observe", req_args))
        runtime = RobotHarnessRuntime(MuJoCoBackend())
        try:
            return _print_json(runtime.observe(**req_args))
        finally:
            runtime.close()

    if args.subcmd == "xarm":
        kwargs: dict[str, Any] = json.loads(args.args)
        for key in ("x", "y", "z", "roll", "pitch", "yaw", "speed", "pos"):
            val = getattr(args, key)
            if val is not None:
                kwargs[key] = int(val) if (args.method == "set_gripper_g2_position" and key == "pos") else val
        if use_daemon:
            return _print_json(
                client.request(
                    "xarm",
                    {
                        "method": args.method,
                        "kwargs": kwargs,
                        "save_image": args.save_image,
                        "camera": args.camera,
                    },
                )
            )
        runtime = RobotHarnessRuntime(MuJoCoBackend())
        try:
            return _print_json(
                runtime.call_xarm(args.method, save_image=args.save_image, camera=args.camera, **kwargs)
            )
        finally:
            runtime.close()

    if args.subcmd == "act":
        params: dict[str, Any] = json.loads(args.params)
        if args.object_id is not None:
            params["object_id"] = args.object_id
        elif args.primitive == "pick" and "object_id" not in params:
            params["object_id"] = "red_cube"
        if args.target_id is not None:
            params["target_id"] = args.target_id
        elif args.primitive == "place" and "target_id" not in params:
            params["target_id"] = "green_target"

        if use_daemon:
            return _print_json(
                client.request(
                    "act",
                    {
                        "primitive": args.primitive,
                        "params": params,
                        "save_image": not args.no_image,
                        "camera": args.camera,
                    },
                )
            )
        runtime = RobotHarnessRuntime(MuJoCoBackend())
        try:
            return _print_json(
                runtime.act(args.primitive, save_image=not args.no_image, camera=args.camera, **params)
            )
        finally:
            runtime.close()

    if args.subcmd == "reset":
        rst_args = {
            "cube_x": args.cube_x,
            "cube_y": args.cube_y,
            "target_x": args.target_x,
            "target_y": args.target_y,
            "save_image": not args.no_image,
        }
        if use_daemon:
            return _print_json(client.request("reset", rst_args))
        runtime = RobotHarnessRuntime(MuJoCoBackend())
        try:
            return _print_json(runtime.reset_scene(**rst_args))
        finally:
            runtime.close()

    return 1


def _run_oneshot_instruction(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="LLM Pick-and-Place Robot Control")
    parser.add_argument(
        "--instruction",
        default="把红色方块抓到绿色目标区域",
        help="Natural language instruction for the robot",
    )
    parser.add_argument("--cube-x", type=float, default=0.30)
    parser.add_argument("--cube-y", type=float, default=0.00)
    parser.add_argument("--target-x", type=float, default=0.30)
    parser.add_argument("--target-y", type=float, default=0.30)
    parser.add_argument(
        "--viewer",
        action="store_true",
        help="Open interactive MuJoCo 3D viewer window during execution",
    )
    parser.add_argument(
        "--hold",
        action="store_true",
        help="Keep the 3D viewer window open after task completion until closed manually",
    )
    parser.add_argument(
        "--save-video",
        type=Path,
        default=None,
        help="Optional output MP4 file path to record the simulation (e.g. runs/pick_place.mp4)",
    )
    args = parser.parse_args(argv)

    backend = MuJoCoBackend(viewer=args.viewer, record_video_path=args.save_video)
    try:
        backend.set_cube_xy(args.cube_x, args.cube_y)
        backend.set_target_xy(args.target_x, args.target_y)

        offset_z = backend.measure_flange_to_tcp_offset_z(ref_xy=(args.cube_x, args.cube_y))
        height_params = PickPlaceHeightParams(flange_to_tcp_z_m=offset_z)

        perception = SimGTPerception(backend)
        planner = LLMPlanner()
        orchestrator = TaskOrchestrator(backend, perception, planner, height_params)

        report = orchestrator.run_instruction(args.instruction)
        print(f"Instruction : {report.instruction}")
        print(f"Plan        : {[f'{p.name}({p.params})' for p in report.plan]}")
        print(f"Result      : {report.message}")

        if args.viewer and args.hold:
            backend.wait_viewer()
    finally:
        saved_video = backend.close()
        if saved_video is not None:
            print(f"Saved video : {saved_video}")

    return 0 if report.success else 1


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] in SUBCOMMANDS:
        return _run_harness_subcommand(sys.argv[1:])
    return _run_oneshot_instruction(sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
