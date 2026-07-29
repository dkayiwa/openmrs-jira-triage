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
    bot_first_labeled_at: str | None = None  # changelog timestamp of the bot's first ai-label add


def _labels(value: str | None) -> set[str]:
    # Changelog label lists are separated by spaces (observed live) but labels
    # can never contain spaces or commas, so tolerate both separators.
    return set((value or "").replace(",", " ").split())


def inspect(issue: dict, ai_labels: list[str], bot_account_id: str | None,
            histories: list[dict] | None = None) -> TicketState:
    """Walk the changelog's labels-field items.

    With no bot_account_id configured, every ai-triage removal counts as an
    opt-out (the safe direction), and manual-add violations are not attributed.

    `histories` overrides the issue's embedded changelog; callers that must not
    miss an opt-out pass the full paged history (JiraClient.changelog), since
    expand=changelog truncates at 100 entries.
    """
    st = TicketState(
        ai_labels_present=[l for l in issue["fields"].get("labels", []) if l in ai_labels]
    )
    if histories is None:
        histories = (issue.get("changelog") or {}).get("histories", [])
    ai_label_set = set(ai_labels)
    for history in histories:
        author = history.get("author") or {}
        author_id = author.get("accountId")
        is_bot = bot_account_id is not None and author_id == bot_account_id
        for item in history.get("items", []):
            if item.get("field") != "labels":
                continue
            before = _labels(item.get("fromString"))
            after = _labels(item.get("toString"))
            if (before - after) & ai_label_set and not is_bot:
                st.opted_out = True
                st.opted_out_by = author.get("displayName") or author_id
            if (after - before) & ai_label_set:
                if is_bot:
                    # Keep the earliest add regardless of history ordering.
                    # Timestamps within one response share a timezone, so
                    # string comparison orders them correctly.
                    added_at = history.get("created")
                    if added_at and (st.bot_first_labeled_at is None or added_at < st.bot_first_labeled_at):
                        st.bot_first_labeled_at = added_at
                elif bot_account_id is not None:
                    st.human_adds.append(author.get("displayName") or author_id)
    return st
