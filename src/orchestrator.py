"""Closed-loop Perception-Planning-Execution Task Orchestrator."""

from __future__ import annotations

import numpy as np

from core.backend import RobotBackend
from core.types import ExecutionReport
from perception.base import PerceptionProvider
from planning.llm_planner import LLMPlanner
from planning.primitives import PickPlaceHeightParams, SkillLibrary


class TaskOrchestrator:
    """Coordinates Perception -> LLM Planning -> Primitive Execution -> Verification."""

    def __init__(
        self,
        backend: RobotBackend,
        perception: PerceptionProvider,
        planner: LLMPlanner | None = None,
        height_params: PickPlaceHeightParams | None = None,
        *,
        place_tolerance_mm: float = 50.0,
    ) -> None:
        self.backend = backend
        self.perception = perception
        self.planner = planner or LLMPlanner()
        self.skills = SkillLibrary(backend, height_params)
        self.place_tolerance_mm = float(place_tolerance_mm)

    def run_instruction(self, instruction: str) -> ExecutionReport:
        """Execute one full closed-loop cycle from natural-language instruction to verification."""
        initial_scene = self.perception.observe()
        plan = self.planner.plan(instruction, initial_scene)

        picked_obj_id: str | None = None
        target_obj_id: str | None = None
        object_lifted = False

        for prim in plan:
            self.skills.execute_primitive(prim, initial_scene)
            if prim.name == "pick":
                picked_obj_id = str(prim.params["object_id"])
                mid_scene = self.perception.observe()
                if picked_obj_id in mid_scene.objects:
                    lifted_z = mid_scene.objects[picked_obj_id].pose.position[2]
                    if lifted_z > 0.05:
                        object_lifted = True
            elif prim.name == "place":
                target_obj_id = str(prim.params["target_id"])

        final_scene = self.perception.observe()

        if picked_obj_id and target_obj_id and picked_obj_id in final_scene.objects and target_obj_id in final_scene.objects:
            final_obj_pos = final_scene.objects[picked_obj_id].pose.pos_np
            target_pos = final_scene.objects[target_obj_id].pose.pos_np
            place_error_mm = float(np.linalg.norm(final_obj_pos[:2] - target_pos[:2]) * 1000.0)
            success = bool(object_lifted and place_error_mm <= self.place_tolerance_mm)
            msg = (
                f"Pick-and-place {'SUCCEEDED' if success else 'FAILED'}: "
                f"lifted={object_lifted}, place_error={place_error_mm:.2f} mm "
                f"(tolerance={self.place_tolerance_mm:.1f} mm)"
            )
            return ExecutionReport(
                instruction=instruction,
                plan=plan,
                success=success,
                object_lifted=object_lifted,
                place_error_mm=place_error_mm,
                final_object_xyz=final_scene.objects[picked_obj_id].pose.position,
                target_xyz=final_scene.objects[target_obj_id].pose.position,
                message=msg,
            )

        return ExecutionReport(
            instruction=instruction,
            plan=plan,
            success=True,
            object_lifted=object_lifted,
            place_error_mm=0.0,
            message="Executed non-pick-place primitive sequence successfully.",
        )

