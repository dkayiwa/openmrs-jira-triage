"""Skip and opt-out decisions, derived from the issue itself.

The Jira changelog is the state store: a non-bot removal of an ai-triage label
is a permanent opt-out (the design doc's convention), and a non-bot *add*
violates the no-manual-labels convention (surfaced in the weekly digest).
The content hash of the last triage lives in an issue entity property, so no
external database is needed for correctness.
"""
from __future__ import annotations

from dataclasses import dataclass, field

PROPERTY_KEY = "ai-triage"


@dataclass
class TicketState:
    ai_labels_present: list[str] = field(default_factory=list)
    opted_out: bool = False
    opted_out_by: str | None = None
    human_adds: list[str] = field(default_factory=list)  # convention violations


def inspect(issue: dict, ai_labels: list[str], bot_account_id: str | None) -> TicketState:
    """Walk the changelog's labels-field items (space-separated label lists).

    With no bot_account_id configured, every ai-triage removal counts as an
    opt-out (the safe direction), and manual-add violations are not attributed.
    """
    st = TicketState(
        ai_labels_present=[l for l in issue["fields"].get("labels", []) if l in ai_labels]
    )
    for history in (issue.get("changelog") or {}).get("histories", []):
        author = history.get("author") or {}
        author_id = author.get("accountId")
        is_bot = bot_account_id is not None and author_id == bot_account_id
        for item in history.get("items", []):
            if item.get("field") != "labels":
                continue
            before = set((item.get("fromString") or "").split())
            after = set((item.get("toString") or "").split())
            if (before - after) & set(ai_labels) and not is_bot:
                st.opted_out = True
                st.opted_out_by = author.get("displayName") or author_id
            if (after - before) & set(ai_labels) and not is_bot and bot_account_id is not None:
                st.human_adds.append(author.get("displayName") or author_id)
    return st
