"""T2 Simulation Tests: Gripper G2 linkage, 20-trial random Pick-Place stability, and end-to-end Orchestrator."""

from __future__ import annotations

import numpy as np

from execution.ik import gripper_g2_gap_m_to_sim_drive, gripper_g2_sim_drive_to_gap_m
from execution.mujoco_backend import MuJoCoBackend
from orchestrator import TaskOrchestrator
from perception.sim_gt import SimGTPerception
from planning.llm_planner import LLMPlanner
from planning.primitives import PickPlaceHeightParams


def test_gripper_g2_linkage_roundtrip() -> None:
    """Verify Gripper G2 4-bar linkage gap <-> drive angle roundtrip accuracy."""
    for gap_m in np.linspace(0.002, 0.082, 20):
        drive = gripper_g2_gap_m_to_sim_drive(float(gap_m))
        recovered = gripper_g2_sim_drive_to_gap_m(drive)
        assert abs(recovered - float(gap_m)) < 1e-6


def test_pick_place_20_random_trials() -> None:
    """Run 20 randomized cube/target configurations in MuJoCo; require >= 95% success rate."""
    backend = MuJoCoBackend()
    perception = SimGTPerception(backend)
    planner = LLMPlanner(provider="rule_based")
    offset_z = backend.measure_flange_to_tcp_offset_z(ref_xy=(0.30, 0.0))
    height_params = PickPlaceHeightParams(flange_to_tcp_z_m=offset_z, cart_v_m_s=0.22)
    orchestrator = TaskOrchestrator(backend, perception, planner, height_params, place_tolerance_mm=50.0)

    rng = np.random.default_rng(2026)
    n_trials = 20
    successes = 0

    for trial in range(n_trials):
        backend.reset_scene()
        cube_x = float(rng.uniform(0.26, 0.36))
        cube_y = float(rng.uniform(-0.12, 0.05))
        target_x = float(rng.uniform(0.26, 0.36))
        target_y = float(rng.uniform(0.18, 0.32))

        backend.set_cube_xy(cube_x, cube_y)
        backend.set_target_xy(target_x, target_y)

        report = orchestrator.run_instruction("把红色方块抓到绿色目标区域")
        if report.success and report.object_lifted and report.place_error_mm < 50.0:
            successes += 1

    success_rate = successes / n_trials
    assert success_rate >= 0.95, f"Pick-place success rate {success_rate:.1%} < 95% ({successes}/{n_trials})"

