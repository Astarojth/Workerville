from __future__ import annotations

from collections import defaultdict

from benchmark_core.shared.types import Message


class CommunicationSubstrate:
    def __init__(self) -> None:
        self._forum: list[Message] = []
        self._inbox: dict[str, list[Message]] = defaultdict(list)

    def dispatch(self, msg: Message) -> None:
        if msg.channel == "forum":
            self._forum.append(msg)
            return
        if msg.channel in {"internal", "external"}:
            self._forum.append(msg)
            return
        self._inbox[msg.target_id].append(msg)

    def seed_messages(self, messages: list[Message]) -> None:
        for msg in messages:
            self.dispatch(msg)

    def pull_inbox(self, agent_id: str) -> list[Message]:
        items = self._inbox.get(agent_id, [])
        self._inbox[agent_id] = []
        return items

    def forum_recent(self, limit: int = 20) -> list[Message]:
        return self._forum[-limit:]

    def snapshot(self) -> dict[str, list[dict]]:
        return {
            "forum": [m.__dict__ for m in self._forum],
            "inbox": {k: [m.__dict__ for m in v] for k, v in self._inbox.items()},
        }
