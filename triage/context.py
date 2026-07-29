"""Assemble the "visible information" context for one ticket.

The pilot contract says the classifier sees only: summary, description,
acceptance criteria, parent ticket, linked tickets, and human comments.
Enforcing that here in code (rather than asking the model to ignore bot
content) makes the guarantee auditable: the exact text sent to the model is
saved to out/contexts/<KEY>.txt.
"""
from __future__ import annotations

import hashlib

from .jira import JiraClient

ISSUE_FIELDS = ["summary", "description", "labels", "status", "created", "comment", "parent", "issuelinks"]


def discover_ac_field(client: JiraClient, configured: str) -> str | None:
    if configured:
        return configured
    for f in client.fields():
        if f.get("name", "").strip().lower() == "acceptance criteria":
            return f["id"]
    return None


def is_human_comment(comment: dict, blocked_ids: list[str]) -> bool:
    author = comment.get("author") or {}
    return author.get("accountType") != "app" and author.get("accountId") not in blocked_ids


def assemble(client: JiraClient, issue: dict, ac_field: str | None, blocked_ids: list[str]) -> str:
    f = issue["fields"]
    lines = [f"TICKET: {issue['key']}", f"SUMMARY: {f.get('summary') or ''}", ""]
    lines += ["DESCRIPTION:", (f.get("description") or "(empty)").strip(), ""]
    if ac_field:
        ac = f.get(ac_field)
        lines += ["ACCEPTANCE CRITERIA:", (str(ac).strip() if ac else "(empty)"), ""]
    parent = f.get("parent")
    if parent:
        try:
            p = client.issue(parent["key"], ["summary", "description"])
            lines += [
                f"PARENT {parent['key']}: {p['fields'].get('summary') or ''}",
                (p["fields"].get("description") or "(empty)").strip(),
                "",
            ]
        except Exception:
            # Keep the audit trail honest: the model did not see the parent body.
            lines += [f"PARENT {parent['key']}: {parent.get('fields', {}).get('summary', '')}",
                      "(parent description unavailable)", ""]
    links = []
    for link in f.get("issuelinks") or []:
        if "outwardIssue" in link:
            other, rel = link["outwardIssue"], link["type"]["outward"]
        elif "inwardIssue" in link:
            other, rel = link["inwardIssue"], link["type"]["inward"]
        else:
            continue
        links.append(f"- {rel} {other['key']}: {other.get('fields', {}).get('summary', '')}")
    if links:
        lines += ["LINKED TICKETS:"] + links + [""]
    all_comments = client.comments(issue["key"], f.get("comment"))
    humans = [c for c in all_comments if is_human_comment(c, blocked_ids)]
    if humans:
        lines.append("HUMAN COMMENTS:")
        for c in humans:
            author = (c.get("author") or {}).get("displayName", "unknown")
            lines += [f"--- {author}, {c.get('created', '')}", (c.get("body") or "").strip()]
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def content_hash(context: str) -> str:
    return hashlib.sha256(context.encode()).hexdigest()[:16]
