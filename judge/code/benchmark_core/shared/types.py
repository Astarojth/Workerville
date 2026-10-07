from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class Message:
    sender_id: str
    target_id: str
    channel: str  # dm | forum | internal | external
    content: str
    timestamp: str = field(default_factory=lambda: utc_now())


@dataclass
class ToolRequest:
    name: str
    args: dict[str, Any]


@dataclass
class ToolResult:
    ok: bool
    output: dict[str, Any] = field(default_factory=dict)
    side_effects: list[str] = field(default_factory=list)
    error: str | None = None


@dataclass
class Decision:
    messages: list[Message] = field(default_factory=list)
    tool_requests: list[ToolRequest] = field(default_factory=list)
    memory_writes: list[str] = field(default_factory=list)
    action_summary: str = ""
    continue_turn: bool = False


@dataclass
class TaskSpec:
    task_id: str
    instruction: str


@dataclass
class AgentSpec:
    agent_id: str
    kind: str
    role: str
    m1: str = "neu"
    permissions: list[str] = field(default_factory=list)
    owner_id: str = "owner_1"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentContext:
    episode_id: str
    step_id: int
    agent: AgentSpec
    task: TaskSpec
    inbox: list[Message]
    forum_recent: list[Message]
    memory: list[str]
    shared_state: dict[str, Any]
    seed: int
    openclaw_profile: dict[str, Any] = field(default_factory=dict)
    turn_feedback: list[dict[str, Any]] = field(default_factory=list)
    subturn_id: int = 0


@dataclass
class LogRecord:
    episode_id: str
    step_id: int
    timestamp: str
    agent_id: str
    role: str
    observation: dict[str, Any]
    action: str
    tool_calls: list[dict[str, Any]]
    messages: list[dict[str, Any]]
    channel: str = "mixed"
    target_id: str = ""
    policy_decision: list[dict[str, Any]] = field(default_factory=list)
    memory_events: list[dict[str, Any]] = field(default_factory=list)
    memory_before_hash: str = ""
    memory_after_hash: str = ""
    seed: int = 0
    config_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EpisodeConfig:
    experiment_name: str
    episode_id: str
    steps: int
    seed: int
    task: TaskSpec
    agents: list[AgentSpec]
    mechanisms: dict[str, bool]
    resources: dict[str, Any]
    runtime: dict[str, Any] = field(default_factory=dict)
    llm: dict[str, Any] = field(default_factory=dict)
    initial_messages: list[dict[str, Any]] = field(default_factory=list)
    timeline_events: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class EpisodeResult:
    logs: list[dict[str, Any]]
    metrics: dict[str, Any]
    final_state: dict[str, Any]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
