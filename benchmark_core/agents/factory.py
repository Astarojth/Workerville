from __future__ import annotations

from benchmark_core.agents.base import Agent
from benchmark_core.agents.social_llm_agent import SocialLLMAgent
from benchmark_core.runtime.api_config import APISettings
from benchmark_core.shared.types import AgentSpec
from agent_os.openclaw_adapter import OpenClawAdapter
from agent_os.openclaw_native_agent import OpenClawNativeAgent
from agent_os.openclaw_summarizer_agent import OpenClawSummarizerAgent


def build_agent(
    spec: AgentSpec,
    runtime_cfg: dict | None = None,
    openclaw: OpenClawAdapter | None = None,
    api_settings: APISettings | None = None,
) -> Agent:
    metadata = spec.metadata if isinstance(spec.metadata, dict) else {}
    driver = str(metadata.get("agent_driver", "")).strip().lower()
    if not driver:
        driver = "openclaw" if spec.kind == "worker" else "llm_chat"

    if driver == "llm_chat":
        if api_settings is None:
            raise ValueError("api_settings is required for llm_chat social agents")
        return SocialLLMAgent(spec.agent_id, api_settings=api_settings)

    if openclaw is None or not openclaw.is_enabled():
        raise ValueError("OpenClaw adapter is required for openclaw-driven agents")
    runtime_cfg = runtime_cfg if isinstance(runtime_cfg, dict) else {}
    openclaw_cfg = runtime_cfg.get("openclaw", {}) if isinstance(runtime_cfg.get("openclaw", {}), dict) else {}
    # Keep turn timeout explicit to prevent silent hangs in long adversarial episodes.
    turn_timeout_sec = int(openclaw_cfg.get("agent_turn_timeout_sec", 120))
    openclaw_module = str(metadata.get("openclaw_module", "")).strip().lower()
    if spec.role == "summarizer" or openclaw_module == "summarize":
        return OpenClawSummarizerAgent(
            spec.agent_id,
            kind=spec.kind,
            openclaw=openclaw,
            turn_timeout_sec=turn_timeout_sec,
        )
    return OpenClawNativeAgent(
        spec.agent_id,
        kind=spec.kind,
        openclaw=openclaw,
        turn_timeout_sec=turn_timeout_sec,
    )
