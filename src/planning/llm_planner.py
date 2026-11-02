"""LLM Task Planner with Tool Calling schema validation and deterministic fallback."""

from __future__ import annotations

import json
import os
import re
from typing import Any

import httpx

from core.types import ActionPrimitive, SceneState


class PlanValidationError(ValueError):
    """Raised when an LLM output or action plan fails schema or scene validation."""


TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "pick",
            "description": "Pick up a graspable object from the workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "object_id": {
                        "type": "string",
                        "description": "ID of the graspable object in SceneState.objects.",
                    }
                },
                "required": ["object_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "place",
            "description": "Place the currently grasped object onto a target zone.",
            "parameters": {
                "type": "object",
                "properties": {
                    "target_id": {
                        "type": "string",
                        "description": "ID of the target placement zone in SceneState.objects.",
                    }
                },
                "required": ["target_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "move_home",
            "description": "Return the robot arm to its safe home/observation configuration.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]

SYSTEM_PROMPT = """You are a robotic manipulation planner for a UFACTORY xArm 6 with Gripper G2.
Given the current SceneState JSON and a user instruction, output a JSON object with a single key "plan"
containing an ordered list of action primitives.
Allowed primitives:
1. {"name": "pick", "params": {"object_id": "<existing_graspable_object_id>"}}
2. {"name": "place", "params": {"target_id": "<existing_target_object_id>"}}
3. {"name": "move_home", "params": {}}
Never invent object IDs that are not present in SceneState.objects. Always finish pick-and-place tasks with move_home.
"""


class LLMPlanner:
    """Decomposes natural-language instructions + SceneState into validated ActionPrimitives."""

    def __init__(
        self,
        *,
        provider: str = "auto",
        api_base: str | None = None,
        api_key: str | None = None,
        model: str = "gpt-4o-mini",
        timeout_s: float = 15.0,
    ) -> None:
        self.provider = provider
        self.api_base = (api_base or os.environ.get("LLM_API_BASE", "https://api.openai.com/v1")).rstrip("/")
        self.api_key = api_key or os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
        self.model = os.environ.get("LLM_MODEL", model)
        self.timeout_s = float(timeout_s)

    def plan(self, instruction: str, scene: SceneState) -> list[ActionPrimitive]:
        """Generate and strictly validate a sequence of ActionPrimitives."""
        if not instruction or not instruction.strip():
            raise PlanValidationError("Instruction cannot be empty")

        if self.provider == "openai_compatible" or (self.provider == "auto" and self.api_key):
            raw_plan = self._plan_via_llm_api(instruction, scene)
        else:
            raw_plan = self._plan_deterministic(instruction, scene)

        self.validate_plan(raw_plan, scene)
        return raw_plan

    def parse_json_plan(self, raw_json: str, scene: SceneState) -> list[ActionPrimitive]:
        """Parse an LLM JSON response string and validate it against SceneState."""
        try:
            payload = json.loads(raw_json)
        except json.JSONDecodeError as exc:
            raise PlanValidationError(f"Invalid JSON from LLM: {exc}") from exc

        if isinstance(payload, dict) and "plan" in payload:
            items = payload["plan"]
        elif isinstance(payload, list):
            items = payload
        else:
            raise PlanValidationError("LLM JSON must be a list or an object containing 'plan'")

        if not isinstance(items, list) or len(items) == 0:
            raise PlanValidationError("Plan must be a non-empty list of action primitives")

        primitives: list[ActionPrimitive] = []
        for idx, item in enumerate(items):
            if not isinstance(item, dict) or "name" not in item:
                raise PlanValidationError(f"Plan step #{idx} is missing 'name'")
            name = str(item["name"])
            params = item.get("params", {})
            if not isinstance(params, dict):
                raise PlanValidationError(f"Plan step #{idx} 'params' must be a dict")
            primitives.append(ActionPrimitive(name=name, params=dict(params)))

        self.validate_plan(primitives, scene)
        return primitives

    def validate_plan(self, primitives: list[ActionPrimitive], scene: SceneState) -> None:
        """Ensure all primitives and referenced object IDs exist and are valid."""
        if not primitives:
            raise PlanValidationError("Plan cannot be empty")

        allowed_names = {"pick", "place", "move_home", "open_gripper", "close_gripper", "move_tcp"}
        for idx, prim in enumerate(primitives):
            if prim.name not in allowed_names:
                raise PlanValidationError(f"Step #{idx}: unknown primitive {prim.name!r}")

            if prim.name == "pick":
                obj_id = prim.params.get("object_id")
                if not isinstance(obj_id, str) or obj_id not in scene.objects:
                    raise PlanValidationError(
                        f"Step #{idx}: pick references unknown object_id {obj_id!r} "
                        f"(available: {sorted(scene.objects.keys())})"
                    )
                obj = scene.objects[obj_id]
                if obj.category != "graspable":
                    raise PlanValidationError(
                        f"Step #{idx}: object {obj_id!r} has category {obj.category!r}, expected 'graspable'"
                    )

            elif prim.name == "place":
                target_id = prim.params.get("target_id")
                if not isinstance(target_id, str) or target_id not in scene.objects:
                    raise PlanValidationError(
                        f"Step #{idx}: place references unknown target_id {target_id!r} "
                        f"(available: {sorted(scene.objects.keys())})"
                    )

    def _plan_via_llm_api(self, instruction: str, scene: SceneState) -> list[ActionPrimitive]:
        user_msg = json.dumps(
            {
                "instruction": instruction,
                "scene_state": scene.to_prompt_dict(),
            },
            ensure_ascii=False,
        )
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        body = {
            "model": self.model,
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
        }
        with httpx.Client(timeout=self.timeout_s) as client:
            resp = client.post(f"{self.api_base}/chat/completions", headers=headers, json=body)
            resp.raise_for_status()
            data = resp.json()
        content = data["choices"][0]["message"]["content"]
        return self.parse_json_plan(content, scene)

    def _plan_deterministic(self, instruction: str, scene: SceneState) -> list[ActionPrimitive]:
        """Semantic parser for bilingual (Chinese/English) pick-and-place instructions."""
        text = instruction.strip().lower()

        # Check if the instruction explicitly mentions an unknown object_id token like cube_99 / blue_ball
        explicit_tokens = re.findall(r"\b(?:[a-z]+_(?:cube|target|box|ball|object)(?:_\d+)?)\b", text)
        for tok in explicit_tokens:
            if tok not in scene.objects:
                raise PlanValidationError(
                    f"Instruction references non-existent object {tok!r} (available: {sorted(scene.objects.keys())})"
                )

        # Also check color mentions that don't exist in scene
        color_map = {
            "蓝": "blue",
            "blue": "blue",
            "黄": "yellow",
            "yellow": "yellow",
            "紫": "purple",
            "purple": "purple",
        }
        scene_colors = {obj.color.lower() for obj in scene.objects.values() if obj.color}
        for keyword, col in color_map.items():
            if keyword in text and col not in scene_colors:
                raise PlanValidationError(
                    f"Instruction requests {col!r} object which is not present in SceneState (colors: {sorted(scene_colors)})"
                )

        # Home-only command
        if ("回零" in text or "回home" in text or "return home" in text or "move home" in text) and not any(
            k in text for k in ("抓", "拿", "放", "pick", "place", "move red", "把")
        ):
            return [ActionPrimitive(name="move_home", params={})]

        # Match graspable object
        graspables = [obj for obj in scene.objects.values() if obj.category == "graspable"]
        targets = [obj for obj in scene.objects.values() if obj.category == "target_zone"]

        picked_obj = None
        for obj in graspables:
            if obj.object_id.lower() in text or (obj.color and obj.color.lower() in text):
                picked_obj = obj
                break
            if obj.color == "red" and ("红" in text or "方块" in text or "物块" in text or "cube" in text):
                picked_obj = obj
                break
        if picked_obj is None and len(graspables) == 1:
            picked_obj = graspables[0]

        place_target = None
        for tgt in targets:
            if tgt.object_id.lower() in text or (tgt.color and tgt.color.lower() in text):
                place_target = tgt
                break
            if tgt.color == "green" and ("绿" in text or "目标" in text or "区域" in text or "target" in text):
                place_target = tgt
                break
        if place_target is None and len(targets) == 1:
            place_target = targets[0]

        if picked_obj is None or place_target is None:
            raise PlanValidationError(
                f"Cannot resolve graspable object or target zone from instruction: {instruction!r}"
            )

        return [
            ActionPrimitive(name="pick", params={"object_id": picked_obj.object_id}),
            ActionPrimitive(name="place", params={"target_id": place_target.object_id}),
            ActionPrimitive(name="move_home", params={}),
        ]

