from __future__ import annotations

from abc import ABC, abstractmethod

from benchmark_core.shared.types import AgentContext, Decision


class Agent(ABC):
    def __init__(self, agent_id: str):
        self.agent_id = agent_id

    @abstractmethod
    def act(self, ctx: AgentContext) -> Decision:
        raise NotImplementedError
