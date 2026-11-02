"""T1 Unit Tests: LLM Planner instruction parsing, JSON schema validation, and hallucination rejection."""

from __future__ import annotations

import pytest

from core.types import GRIP_DOWN_QUAT_WXYZ, ObjectInfo, Pose, RobotState, SceneState
from planning.llm_planner import LLMPlanner, PlanValidationError


def _make_mock_scene() -> SceneState:
    robot = RobotState(
        joint_positions_rad=(0.0, -0.63, -0.83, 0.0, 1.46, 0.0),
        tcp_pose=Pose.from_xyz(0.30, 0.0, 0.25, GRIP_DOWN_QUAT_WXYZ),
        flange_pose=Pose.from_xyz(0.30, 0.0, 0.4235, GRIP_DOWN_QUAT_WXYZ),
        gripper_gap_m=0.084,
        is_grasping=False,
    )
    objects = {
        "red_cube": ObjectInfo(
            object_id="red_cube",
            category="graspable",
            pose=Pose.from_xyz(0.30, 0.0, 0.015, GRIP_DOWN_QUAT_WXYZ),
            color="red",
        ),
        "green_target": ObjectInfo(
            object_id="green_target",
            category="target_zone",
            pose=Pose.from_xyz(0.30, 0.30, 0.015, GRIP_DOWN_QUAT_WXYZ),
            color="green",
        ),
    }
    return SceneState(robot=robot, objects=objects, timestamp=1000.0)


@pytest.mark.parametrize(
    "instruction",
    [
        "把红色方块抓到绿色目标区域",
        "请抓取 red_cube 并放置到 green_target",
        "Pick up the red cube and place it on the green target zone",
        "把物块放到目标位置",
    ],
)
def test_planner_bilingual_instructions(instruction: str) -> None:
    """Verify 100% parse accuracy across diverse Chinese and English instructions."""
    scene = _make_mock_scene()
    planner = LLMPlanner(provider="rule_based")
    plan = planner.plan(instruction, scene)

    assert len(plan) == 3
    assert plan[0].name == "pick" and plan[0].params == {"object_id": "red_cube"}
    assert plan[1].name == "place" and plan[1].params == {"target_id": "green_target"}
    assert plan[2].name == "move_home"


def test_planner_json_parsing_and_hallucination_guard() -> None:
    """Verify LLM JSON parsing and 100% rejection of hallucinated object IDs or invalid primitives."""
    scene = _make_mock_scene()
    planner = LLMPlanner(provider="rule_based")

    valid_json = """
    {
      "plan": [
        {"name": "pick", "params": {"object_id": "red_cube"}},
        {"name": "place", "params": {"target_id": "green_target"}},
        {"name": "move_home", "params": {}}
      ]
    }
    """
    parsed = planner.parse_json_plan(valid_json, scene)
    assert len(parsed) == 3

    # Hallucinated object_id in JSON
    hallucinated_json = '{"plan": [{"name": "pick", "params": {"object_id": "blue_cube_99"}}]}'
    with pytest.raises(PlanValidationError, match="unknown object_id"):
        planner.parse_json_plan(hallucinated_json, scene)

    # Hallucinated target_id in JSON
    bad_target_json = '{"plan": [{"name": "place", "params": {"target_id": "yellow_zone"}}]}'
    with pytest.raises(PlanValidationError, match="unknown target_id"):
        planner.parse_json_plan(bad_target_json, scene)

    # Attempting to pick a non-graspable target_zone
    wrong_cat_json = '{"plan": [{"name": "pick", "params": {"object_id": "green_target"}}]}'
    with pytest.raises(PlanValidationError, match="expected 'graspable'"):
        planner.parse_json_plan(wrong_cat_json, scene)

    # Hallucinated object in natural-language instruction
    with pytest.raises(PlanValidationError):
        planner.plan("把蓝色方块抓到绿色区域", scene)
    with pytest.raises(PlanValidationError):
        planner.plan("pick blue_cube and place on green_target", scene)

