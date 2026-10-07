from __future__ import annotations

from dataclasses import asdict

from benchmark_core.agents.factory import build_agent
from benchmark_core.agents.m1_model import resolve_m1
from benchmark_core.detectors.metrics import compute_metrics
from benchmark_core.env.world import SandboxWorld
from benchmark_core.runtime import APISettings
from benchmark_core.shared.io import hash_memory, hash_payload
from benchmark_core.shared.types import AgentContext, EpisodeConfig, EpisodeResult, LogRecord, Message, ToolRequest, utc_now


class EpisodeRunner:
    def run(self, config: EpisodeConfig, api_settings: APISettings | None = None) -> EpisodeResult:
        if api_settings is None:
            raise ValueError("api_settings is required: OpenClaw native runtime needs provider/model credentials")
        world = SandboxWorld(
            task=config.task,
            agents=config.agents,
            resources_cfg=config.resources,
            initial_messages_cfg=config.initial_messages,
            runtime_cfg=config.runtime,
            mechanisms_cfg=config.mechanisms,
        )
        llm_overrides = dict(config.llm) if isinstance(config.llm, dict) else {}
        openclaw_runtime = config.runtime.get("openclaw", {}) if isinstance(config.runtime.get("openclaw", {}), dict) else {}
        if "agent_turn_timeout_sec" in openclaw_runtime:
            llm_overrides["gateway_agent_timeout_seconds"] = int(openclaw_runtime["agent_turn_timeout_sec"])
        world.openclaw.sync_gateway_model_config(api_settings, llm_overrides=llm_overrides)
        agents = {
            spec.agent_id: build_agent(
                spec,
                runtime_cfg=config.runtime,
                openclaw=world.openclaw,
                api_settings=api_settings,
            )
            for spec in config.agents
        }

        shared_state = {
            "workers": [spec.agent_id for spec in config.agents if spec.kind == "worker"],
            "peers": [spec.agent_id for spec in config.agents if spec.role == "peer"],
            "summarizers": [spec.agent_id for spec in config.agents if spec.role == "summarizer"],
            "mechanisms": config.mechanisms,
            "runtime": config.runtime,
            "llm": config.llm,
            "environment": self._build_environment_index(world),
        }
        openclaw_cfg = config.runtime.get("openclaw", {}) if isinstance(config.runtime.get("openclaw", {}), dict) else {}
        max_subturns = max(1, int(openclaw_cfg.get("max_subturns", 3)))
        config_hash = hash_payload(asdict(config))
        logs: list[dict] = []
        processed_timeline_events: set[int] = set()

        for step in range(config.steps):
            injected_rows = self._apply_timeline_events(
                timeline_events=config.timeline_events,
                processed_event_ids=processed_timeline_events,
                step=step,
                world=world,
            )
            if injected_rows:
                shared_state["environment"] = self._build_environment_index(world)
                print(
                    f"[Workerville] timeline step={step + 1}: applied={len(injected_rows)}",
                    flush=True,
                )
            print(
                f"[Workerville] episode={config.episode_id} step={step + 1}/{config.steps} agents={len(config.agents)}",
                flush=True,
            )
            for spec in config.agents:
                inbox = world.comms.pull_inbox(spec.agent_id) if config.mechanisms.get("enable_m2", True) else []
                turn_feedback: list[dict] = []
                for subturn in range(max_subturns):
                    forum_recent = world.comms.forum_recent()
                    memory = list(world.memory_store.get(spec.agent_id, []))
                    before_hash = hash_memory(memory)

                    ctx = AgentContext(
                        episode_id=config.episode_id,
                        step_id=step,
                        agent=spec,
                        task=config.task,
                        inbox=inbox,
                        forum_recent=forum_recent,
                        memory=memory,
                        shared_state=shared_state,
                        seed=config.seed,
                        openclaw_profile=world.get_openclaw_profile(spec.agent_id, spec.permissions),
                        turn_feedback=turn_feedback,
                        subturn_id=subturn,
                    )
                    decision = agents[spec.agent_id].act(ctx)
                    tool_calls, policy_rows, secret_cache, feedback_items = self._execute_tools(
                        spec,
                        decision.tool_requests,
                        world,
                        allowed_tools=self._allowed_tools(spec, ctx.openclaw_profile),
                    )
                    msgs = self._dispatch_messages(decision.messages, world, secret_cache, config)
                    memory_events = self._apply_memory_writes(spec, decision.memory_writes, world, config)
                    m1_effective, m1_score = resolve_m1(
                        memory,
                        m1_enabled=bool(config.mechanisms.get("enable_m1", True)),
                        fallback_m1=spec.m1,
                    )

                    obs = {
                        "task": config.task.instruction,
                        "inbox": [m.__dict__ for m in inbox],
                        "forum_recent_count": len(forum_recent),
                        "memory_size": len(memory),
                        "memory_profile": str(spec.metadata.get("memory_profile", "")) if isinstance(spec.metadata, dict) else "",
                        "memory_context_lines": int(spec.metadata.get("memory_context_lines", 0))
                        if isinstance(spec.metadata, dict) and str(spec.metadata.get("memory_context_lines", "")).isdigit()
                        else 0,
                        "m1_effective": m1_effective,
                        "m1_score": m1_score,
                        "openclaw_enabled": bool(ctx.openclaw_profile.get("enabled", False))
                        if isinstance(ctx.openclaw_profile, dict)
                        else False,
                        "openclaw_skills_count": len(ctx.openclaw_profile.get("skills", []))
                        if isinstance(ctx.openclaw_profile, dict) and isinstance(ctx.openclaw_profile.get("skills", []), list)
                        else 0,
                        "subturn_id": subturn,
                        "turn_feedback_count": len(turn_feedback),
                    }
                    world.append_raw_log(spec.agent_id, f"step={step} subturn={subturn} action={decision.action_summary}")

                    after_hash = hash_memory(world.memory_store.get(spec.agent_id, []))
                    row = LogRecord(
                        episode_id=config.episode_id,
                        step_id=step,
                        timestamp=utc_now(),
                        agent_id=spec.agent_id,
                        role=spec.role,
                        observation=obs,
                        action=decision.action_summary,
                        tool_calls=tool_calls,
                        messages=msgs,
                        channel="mixed",
                        target_id=msgs[0].get("target_id", "") if msgs else "",
                        policy_decision=policy_rows,
                        memory_events=memory_events,
                        memory_before_hash=before_hash,
                        memory_after_hash=after_hash,
                        seed=config.seed,
                        config_hash=config_hash,
                    ).to_dict()
                    logs.append(row)
                    self._update_worker_event_window(shared_state=shared_state, row=row)
                    if feedback_items:
                        turn_feedback.extend(feedback_items)
                    inbox = []

                    should_continue = (
                        bool(decision.continue_turn)
                        and bool(tool_calls)
                        and subturn + 1 < max_subturns
                    )
                    if not should_continue:
                        break

        metrics = compute_metrics(logs, list(world.resources.secrets.values()))
        return EpisodeResult(logs=logs, metrics=metrics, final_state=world.snapshot())

    @staticmethod
    def _update_worker_event_window(shared_state: dict, row: dict) -> None:
        if str(row.get("role", "")).strip().lower() != "worker":
            return
        tool_calls = row.get("tool_calls", [])
        if not isinstance(tool_calls, list):
            tool_calls = []
        tools: list[str] = []
        risk_tags: set[str] = set()
        for call in tool_calls:
            if not isinstance(call, dict):
                continue
            name = str(call.get("name", "")).strip()
            if name:
                tools.append(name)
            for tag in call.get("risk_tags", []) if isinstance(call.get("risk_tags", []), list) else []:
                text = str(tag).strip()
                if text:
                    risk_tags.add(text)
            policy = call.get("policy", {}) if isinstance(call.get("policy", {}), dict) else {}
            for tag in policy.get("risk_tags", []) if isinstance(policy.get("risk_tags", []), list) else []:
                text = str(tag).strip()
                if text:
                    risk_tags.add(text)
        event = {
            "step_id": int(row.get("step_id", 0)),
            "agent_id": str(row.get("agent_id", "")),
            "action": str(row.get("action", ""))[:220],
            "tools": tools[:6],
            "risk_tags": sorted(risk_tags),
        }
        window = shared_state.setdefault("worker_event_window", [])
        if not isinstance(window, list):
            window = []
            shared_state["worker_event_window"] = window
        window.append(event)
        if len(window) > 24:
            del window[:-24]

    @staticmethod
    def _apply_timeline_events(
        timeline_events: list[dict],
        processed_event_ids: set[int],
        step: int,
        world,
    ) -> list[dict[str, str]]:
        if not isinstance(timeline_events, list):
            return []
        applied: list[dict[str, str]] = []
        for idx, event in enumerate(timeline_events):
            if idx in processed_event_ids:
                continue
            if not isinstance(event, dict):
                continue
            event_step = int(event.get("step", -1))
            if event_step not in {step, step + 1}:
                continue
            event_type = str(event.get("type", "message")).strip().lower()
            if event_type == "message":
                msg = Message(
                    sender_id=str(event.get("from", "env_timeline")).strip() or "env_timeline",
                    target_id=str(event.get("to", "")).strip(),
                    channel=str(event.get("channel", "dm")).strip() or "dm",
                    content=str(event.get("content", "")),
                )
                world.comms.dispatch(msg)
            elif event_type == "memory_write":
                target = str(event.get("target", "")).strip()
                line = str(event.get("content", "")).strip()
                if target and line:
                    world.append_memory(target, line)
            elif event_type == "db_insert":
                world.resources.db_insert(
                    database=str(event.get("database", "")).strip(),
                    table=str(event.get("table", "")).strip(),
                    row=event.get("row", {}) if isinstance(event.get("row", {}), dict) else {},
                    actor=str(event.get("actor", "timeline")),
                )
            elif event_type == "db_update":
                world.resources.db_update(
                    database=str(event.get("database", "")).strip(),
                    table=str(event.get("table", "")).strip(),
                    filters=event.get("filters", {}) if isinstance(event.get("filters", {}), dict) else {},
                    updates=event.get("updates", {}) if isinstance(event.get("updates", {}), dict) else {},
                    actor=str(event.get("actor", "timeline")),
                )
            elif event_type == "file_write":
                path = str(event.get("path", "")).strip()
                if path:
                    world.resources.write_file(
                        path=path,
                        content=str(event.get("content", "")),
                        author=str(event.get("author", "timeline")),
                    )
            elif event_type == "approval_review":
                world.resources.review_approval(
                    request_id=str(event.get("request_id", "")).strip(),
                    reviewer=str(event.get("reviewer", "owner_1")),
                    decision=str(event.get("decision", "rejected")),
                    note=str(event.get("note", "")),
                )
            else:
                continue
            processed_event_ids.add(idx)
            applied.append({"event_type": event_type, "step": str(step), "idx": str(idx)})
        return applied

    @staticmethod
    def _execute_tools(
        spec,
        requests: list[ToolRequest],
        world,
        allowed_tools: set[str],
    ) -> tuple[list[dict], list[dict], dict[str, str], list[dict[str, str]]]:
        calls = []
        policy_rows = []
        secret_cache: dict[str, str] = {}
        feedback_rows: list[dict[str, str]] = []
        for req in requests:
            effective_req = EpisodeRunner._normalize_tool_request(req, allowed_tools)
            before = EpisodeRunner._resource_probe(world)
            if effective_req.name not in allowed_tools:
                policy = {"decision": "deny", "reason": "permission_denied"}
                result = {"ok": False, "error": "permission_denied", "output": {}}
            else:
                tool_result, policy = world.tools.call(effective_req)
                result = {"ok": tool_result.ok, "error": tool_result.error, "output": tool_result.output}
                if effective_req.name == "read_secret" and tool_result.ok:
                    key_id = str(effective_req.args.get("key_id", ""))
                    secret_cache[key_id] = str(tool_result.output.get("value", ""))
            after = EpisodeRunner._resource_probe(world)
            state_delta = EpisodeRunner._resource_delta(before, after)
            policy_risk_tags = policy.get("risk_tags", []) if isinstance(policy, dict) else []
            if not isinstance(policy_risk_tags, list):
                policy_risk_tags = []
            risk_tags = [str(x) for x in policy_risk_tags if str(x).strip()]
            if effective_req.name == "read_secret":
                risk_tags.append("secret_operation")
            if effective_req.name in {"transfer", "funds_transfer_workflow"}:
                risk_tags.append("financial_operation")
            policy_rows.append(policy)
            call_row = {
                "name": effective_req.name,
                "args": effective_req.args,
                "result": result,
                "policy": policy,
                "request": {"name": effective_req.name, "args": effective_req.args},
                "response": result,
                "state_before": before,
                "state_after": after,
                "state_delta": state_delta,
                "risk_tags": sorted(set(risk_tags)),
            }
            if effective_req.name != req.name:
                call_row["original_request"] = {"name": req.name, "args": req.args}
            calls.append(call_row)
            feedback_rows.append(
                {
                    "tool": str(effective_req.name),
                    "ok": str(bool(result.get("ok", False))).lower(),
                    "error": str(result.get("error", ""))[:200],
                    "output_preview": EpisodeRunner._output_preview(result.get("output", {})),
                    "policy": str(policy.get("decision", "")) if isinstance(policy, dict) else "",
                    "risk_tags": ",".join(sorted(set(risk_tags))),
                    "delta": ",".join(state_delta[:6]),
                }
            )
        return calls, policy_rows, secret_cache, feedback_rows

    @staticmethod
    def _normalize_tool_request(req: ToolRequest, allowed_tools: set[str]) -> ToolRequest:
        name = str(req.name).strip()
        if not name or name in allowed_tools:
            return req
        if "run_skill" not in allowed_tools:
            return req
        workflow_name = name.lower()
        if not workflow_name.endswith("_workflow"):
            return req
        args = req.args if isinstance(req.args, dict) else {}
        if not any(k in args for k in ("skill_name", "workflow_name", "workflow_id", "workflow", "task", "name")):
            return ToolRequest(name="run_skill", args={"skill_name": workflow_name, **args})
        return ToolRequest(name="run_skill", args=args)

    @staticmethod
    def _allowed_tools(spec, openclaw_profile: dict) -> set[str]:
        declared = {str(x) for x in spec.permissions}
        if not isinstance(openclaw_profile, dict):
            return declared
        if not bool(openclaw_profile.get("enabled", False)):
            return declared
        skills = openclaw_profile.get("skills", [])
        if not isinstance(skills, list) or not skills:
            return declared
        return {str(x) for x in skills if str(x).strip()}

    @staticmethod
    def _dispatch_messages(messages: list[Message], world, secret_cache: dict[str, str], config: EpisodeConfig) -> list[dict]:
        rows = []
        for msg in messages:
            if not config.mechanisms.get("enable_m2", True) and msg.channel == "dm":
                continue
            content = msg.content
            for key_id, value in secret_cache.items():
                content = content.replace(f"{{{{{key_id}}}}}", value)
            dispatched = Message(
                sender_id=msg.sender_id,
                target_id=msg.target_id,
                channel=msg.channel,
                content=content,
            )
            world.comms.dispatch(dispatched)
            rows.append(dispatched.__dict__)
        return rows

    @staticmethod
    def _apply_memory_writes(spec, writes: list[str], world, config: EpisodeConfig) -> list[dict]:
        events: list[dict] = []
        if not config.mechanisms.get("enable_m3", True):
            return events
        for line in writes:
            if "target=" not in line or "|" not in line:
                world.append_memory(spec.agent_id, line)
                events.append({"source": spec.agent_id, "target": spec.agent_id, "content": line})
                continue
            meta, payload = line.split("|", 1)
            target = meta.replace("target=", "").strip()
            world.append_memory(target, payload)
            events.append({"source": spec.agent_id, "target": target, "content": payload})
        return events

    @staticmethod
    def _build_environment_index(world) -> dict:
        db_schema: dict[str, list[str]] = {}
        for db_name, tables in world.resources.virtual_dbs.items():
            db_schema[str(db_name)] = sorted(str(table) for table in tables.keys())
        identity_roles: dict[str, str] = {}
        for row in world.resources.virtual_dbs.get("identity_db", {}).get("users", []):
            if not isinstance(row, dict):
                continue
            user_id = str(row.get("user_id", "")).strip()
            role = str(row.get("role", "")).strip()
            if user_id and role:
                identity_roles[user_id] = role
        approval_scopes = [
            str(x.get("scope", ""))
            for x in world.resources.virtual_dbs.get("policy_db", {}).get("approval_tickets", [])
            if isinstance(x, dict)
        ][:12]
        pending_approval_count = sum(
            1
            for x in world.resources.virtual_dbs.get("policy_db", {}).get("approval_tickets", [])
            if isinstance(x, dict) and str(x.get("status", "")).strip().lower() != "approved"
        )
        security_policy_titles = [
            str(x.get("title", ""))
            for x in world.resources.virtual_dbs.get("policy_db", {}).get("security_policies", [])
            if isinstance(x, dict)
        ][:12]
        open_incidents_count = sum(
            1
            for x in world.resources.virtual_dbs.get("ops_db", {}).get("incidents", [])
            if isinstance(x, dict) and str(x.get("status", "open")).strip().lower() != "closed"
        )
        active_campaign_count = sum(
            1
            for x in world.resources.virtual_dbs.get("ops_db", {}).get("peer_campaigns", [])
            if isinstance(x, dict) and str(x.get("status", "active")).strip().lower() == "active"
        )
        role_relationships = [
            {
                "from_role": str(x.get("from_role", "")),
                "to_role": str(x.get("to_role", "")),
                "link": str(x.get("link", "")),
            }
            for x in world.resources.virtual_dbs.get("identity_db", {}).get("role_relationships", [])
            if isinstance(x, dict)
        ][:12]
        return {
            "secret_key_ids": sorted(world.resources.secrets.keys()),
            "wallet_accounts": sorted(world.resources.wallets.keys()),
            "file_paths": sorted(world.resources.files.keys()),
            "db_schema": db_schema,
            "ticket_ids": sorted(world.resources.tickets.keys()),
            "approval_ids": sorted(world.resources.approval_requests.keys()),
            "recent_skill_runs_count": len(world.resources.skill_runs),
            "identity_roles": identity_roles,
            "approval_scopes": approval_scopes,
            "pending_approval_count": pending_approval_count,
            "security_policy_titles": security_policy_titles,
            "open_incidents_count": open_incidents_count,
            "active_peer_campaign_count": active_campaign_count,
            "role_relationships": role_relationships,
        }

    @staticmethod
    def _resource_probe(world) -> dict[str, object]:
        return {
            "wallets": {k: float(v) for k, v in world.resources.wallets.items()},
            "tickets_count": len(world.resources.tickets),
            "emails_count": len(world.resources.emails),
            "notes_count": len(world.resources.notes),
            "db_ops_count": len([x for x in world.resources.operation_log if str(x.get("type", "")).startswith("db_")]),
            "approvals_count": len(world.resources.approval_requests),
            "skill_runs_count": len(world.resources.skill_runs),
        }

    @staticmethod
    def _resource_delta(before: dict[str, object], after: dict[str, object]) -> list[str]:
        deltas: list[str] = []
        before_wallets = before.get("wallets", {})
        after_wallets = after.get("wallets", {})
        if isinstance(before_wallets, dict) and isinstance(after_wallets, dict):
            keys = set(before_wallets.keys()) | set(after_wallets.keys())
            for key in sorted(keys):
                b = float(before_wallets.get(key, 0.0))
                a = float(after_wallets.get(key, 0.0))
                if abs(a - b) > 1e-9:
                    deltas.append(f"wallet:{key}:{b}->{a}")
        for key in ("tickets_count", "emails_count", "notes_count", "db_ops_count", "approvals_count", "skill_runs_count"):
            b = int(before.get(key, 0) or 0)
            a = int(after.get(key, 0) or 0)
            if a != b:
                deltas.append(f"{key}:{b}->{a}")
        return deltas

    @staticmethod
    def _output_preview(output: object) -> str:
        text = str(output)
        text = text.replace("\n", " ").replace("\r", " ").strip()
        if len(text) > 320:
            return text[:320] + "...(truncated)"
        return text
