"""Abstract perception interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from core.types import SceneState


class PerceptionProvider(ABC):
    """Produces a structured SceneState in robot base frame."""

    @abstractmethod
    def observe(self) -> SceneState:
        """Capture sensor data and return current SceneState."""

