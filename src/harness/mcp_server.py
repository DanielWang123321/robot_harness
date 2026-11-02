"""Standard MCP (Model Context Protocol) stdio JSON-RPC 2.0 Server for Robot Control Harness."""

from __future__ import annotations

import json
import sys
from typing import Any

from execution.mujoco_backend import MuJoCoBackend

from .runtime import RobotHarnessRuntime
from .session import DEFAULT_SOCKET_PATH, SessionClient

MCP_TOOLS: list[dict[str, Any]] = [
    {
        "name": "xarm_observe",
        "description": "Observe current robot telemetry (TCP mm/deg, joint angles, gripper counts/gap, contact forces N) and 3D object poses in workspace, optionally saving an RGB snapshot.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "save_image": {"type": "boolean", "default": True},
                "camera": {"type": "string", "enum": ["overview", "eye_in_hand"], "default": "overview"},
            },
        },
    },
    {
        "name": "xarm_get_position",
        "description": "XArmAPI.get_position(is_radian=False) -> (code, [x_mm, y_mm, z_mm, roll, pitch, yaw]).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "is_radian": {"type": "boolean", "default": False},
            },
        },
    },
    {
        "name": "xarm_set_position",
        "description": "XArmAPI.set_position(x, y, z, roll, pitch, yaw, speed, relative, is_radian, wait) -> move TCP in mm and deg/rad with workspace & collision guardrails.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "x": {"type": "number", "description": "Target TCP X in mm"},
                "y": {"type": "number", "description": "Target TCP Y in mm"},
                "z": {"type": "number", "description": "Target TCP Z in mm (must be >= 8 mm)"},
                "roll": {"type": "number", "description": "Roll angle (deg by default, 180 for top-down)"},
                "pitch": {"type": "number", "description": "Pitch angle (deg by default, 0 for top-down)"},
                "yaw": {"type": "number", "description": "Yaw angle (deg by default)"},
                "speed": {"type": "number", "description": "Cartesian speed in mm/s (default 150)"},
                "relative": {"type": "boolean", "default": False},
                "is_radian": {"type": "boolean", "default": False},
                "wait": {"type": "boolean", "default": True},
            },
        },
    },
    {
        "name": "xarm_get_servo_angle",
        "description": "XArmAPI.get_servo_angle(servo_id=None, is_radian=False) -> (code, [j1..j6]).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "servo_id": {"type": "integer", "description": "Optional joint index 1..6"},
                "is_radian": {"type": "boolean", "default": False},
            },
        },
    },
    {
        "name": "xarm_set_servo_angle",
        "description": "XArmAPI.set_servo_angle(angle, speed, is_radian, wait) -> move joints with collision & limit checks.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "angle": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "6 joint angles in deg (or rad if is_radian=True)",
                },
                "speed": {"type": "number"},
                "is_radian": {"type": "boolean", "default": False},
                "wait": {"type": "boolean", "default": True},
            },
            "required": ["angle"],
        },
    },
    {
        "name": "xarm_set_gripper_position",
        "description": "XArmAPI.set_gripper_position(pos, wait=True) -> control Gripper G2 in SDK pulse counts [0..840].",
        "inputSchema": {
            "type": "object",
            "properties": {
                "pos": {"type": "number", "description": "Gripper pulse position 0 (closed) .. 840 (open 84mm)"},
                "wait": {"type": "boolean", "default": True},
            },
            "required": ["pos"],
        },
    },
    {
        "name": "xarm_set_gripper_g2_position",
        "description": "XArmAPI.set_gripper_g2_position(pos, speed=100, force=50, wait=True) -> control Gripper G2 gap in mm [0..84].",
        "inputSchema": {
            "type": "object",
            "properties": {
                "pos": {"type": "integer", "description": "Target physical finger gap in mm [0..84]"},
                "speed": {"type": "integer", "default": 100},
                "force": {"type": "integer", "default": 50},
                "wait": {"type": "boolean", "default": True},
            },
            "required": ["pos"],
        },
    },
    {
        "name": "xarm_reset",
        "description": "XArmAPI.reset(wait=True) -> move robot arm back to safe home configuration.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "wait": {"type": "boolean", "default": True},
            },
        },
    },
    {
        "name": "xarm_pick",
        "description": "High-level skill: pick up a graspable object (e.g. 'red_cube') and verify bilateral finger contact.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "object_id": {"type": "string", "default": "red_cube"},
            },
            "required": ["object_id"],
        },
    },
    {
        "name": "xarm_place",
        "description": "High-level skill: place the held object onto a target zone (e.g. 'green_target').",
        "inputSchema": {
            "type": "object",
            "properties": {
                "target_id": {"type": "string", "default": "green_target"},
            },
            "required": ["target_id"],
        },
    },
    {
        "name": "xarm_reset_scene",
        "description": "Reset simulation scene props (cube_x, cube_y, target_x, target_y) and return arm to home.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "cube_x": {"type": "number", "default": 0.30},
                "cube_y": {"type": "number", "default": 0.00},
                "target_x": {"type": "number", "default": 0.30},
                "target_y": {"type": "number", "default": 0.30},
            },
        },
    },
]


class MCPServer:
    """MCP JSON-RPC 2.0 dispatcher routing tool calls to SessionDaemon or in-process Runtime."""

    def __init__(self, runtime: RobotHarnessRuntime | None = None) -> None:
        self._runtime = runtime
        self._client = SessionClient(DEFAULT_SOCKET_PATH)

    def _get_runtime(self) -> RobotHarnessRuntime:
        if self._runtime is None:
            self._runtime = RobotHarnessRuntime(MuJoCoBackend())
        return self._runtime

    def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        args = dict(arguments or {})
        use_daemon = self._runtime is None and self._client.is_running()

        if name == "xarm_observe":
            return self._client.request("observe", args) if use_daemon else self._get_runtime().observe(**args)
        if name == "xarm_pick":
            obj_id = str(args.get("object_id", "red_cube"))
            return (
                self._client.request("act", {"primitive": "pick", "params": {"object_id": obj_id}})
                if use_daemon
                else self._get_runtime().act("pick", object_id=obj_id)
            )
        if name == "xarm_place":
            tgt_id = str(args.get("target_id", "green_target"))
            return (
                self._client.request("act", {"primitive": "place", "params": {"target_id": tgt_id}})
                if use_daemon
                else self._get_runtime().act("place", target_id=tgt_id)
            )
        if name == "xarm_reset_scene":
            return self._client.request("reset", args) if use_daemon else self._get_runtime().reset_scene(**args)

        if name.startswith("xarm_"):
            method = name[len("xarm_") :]
            if use_daemon:
                return self._client.request("xarm", {"method": method, "kwargs": args})
            return self._get_runtime().call_xarm(method, **args)

        raise ValueError(f"Unknown MCP tool: {name!r}")

    def handle_jsonrpc(self, msg: dict[str, Any]) -> dict[str, Any] | None:
        method = msg.get("method")
        msg_id = msg.get("id")
        params = msg.get("params") or {}

        if method == "notifications/initialized":
            return None
        if method == "ping":
            return {"jsonrpc": "2.0", "id": msg_id, "result": {}}
        if method == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "xarm-robot-harness", "version": "0.1.0"},
                },
            }
        if method == "tools/list":
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {"tools": MCP_TOOLS},
            }
        if method == "tools/call":
            tool_name = str(params.get("name", ""))
            tool_args = params.get("arguments") or {}
            try:
                out = self.call_tool(tool_name, tool_args)
                is_error = bool(out.get("code", 0) != 0)
                return {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": {
                        "content": [{"type": "text", "text": json.dumps(out, ensure_ascii=False, indent=2)}],
                        "isError": is_error,
                    },
                }
            except Exception as exc:  # noqa: BLE001
                return {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": {
                        "content": [{"type": "text", "text": json.dumps({"error": str(exc)}, ensure_ascii=False)}],
                        "isError": True,
                    },
                }
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "error": {"code": -32601, "message": f"Method not found: {method}"},
        }

    def serve_stdio(self) -> None:
        """Run MCP stdio loop supporting line-delimited JSON-RPC messages."""
        for line in sys.stdin:
            text = line.strip()
            if not text or text.startswith("Content-Length:"):
                continue
            try:
                msg = json.loads(text)
            except json.JSONDecodeError:
                continue
            resp = self.handle_jsonrpc(msg)
            if resp is not None:
                sys.stdout.write(json.dumps(resp, ensure_ascii=False) + "\n")
                sys.stdout.flush()


if __name__ == "__main__":
    MCPServer().serve_stdio()

