"""Live-phase weekly metrics plus the pre-registered decision rule.

Everything is computed from Jira changelogs alone - the cohort sweep finds
every ticket the bot ever labeled (including later opt-outs), so anyone with
read access and the bot's account id can reproduce the numbers; no local
journal is needed. Thresholds live in config.toml [metrics] and must be
committed before launch - the design doc's "decision rule determined before
results analyzed". The ADOPT/EXTEND/STOP boundaries below are a draft to
pre-register with the pilot owners.

The 24h SLA is measured from max(ticket created, pilot launch): the initial
backlog cohort is older than 24h by definition, so measuring from creation
would fail the metric before the pilot starts.
"""
from __future__ import annotations

import datetime
import os
import sys

from .run import ai_label_names, jira_from_env, load_config
from .state import inspect


def sla_met(created: str, labeled_at: str, launch: datetime.datetime) -> bool:
    """Was the ticket sorted within 24h of entering scope?

    Measured from max(created, launch) because the initial backlog cohort is
    older than 24h by definition.
    """
    start = max(datetime.datetime.fromisoformat(created), launch)
    return datetime.datetime.fromisoformat(labeled_at) - start <= datetime.timedelta(hours=24)


def decide(pct24: float, removal_rate: float, intro: int, m: dict) -> str:
    """The pre-registered decision rule. Committed before launch; never tuned after."""
    passes = [
        pct24 >= m["sorted_within_24h_pct"],
        removal_rate <= m["max_label_removal_rate"],
        intro >= m["min_intro_outcomes"],
    ]
    if all(passes):
        return "ADOPT"
    # Removal rate is the kill metric: past double the threshold, no amount of
    # throughput or intro output earns an extension.
    if removal_rate <= 2 * m["max_label_removal_rate"] and sum(passes) >= 2:
        return "EXTEND (two weeks)"
    return "STOP"


def main() -> int:
    cfg = load_config()
    jira = jira_from_env(cfg)
    bot_id = os.environ.get("TRIAGE_BOT_ACCOUNT_ID")
    if not bot_id:
        sys.exit("TRIAGE_BOT_ACCOUNT_ID is required to attribute the bot's label changes")
    m = cfg["metrics"]
    if not m.get("pilot_launch"):
        sys.exit("set [metrics].pilot_launch in config.toml at launch - it anchors the 24h SLA")
    launch = datetime.datetime.fromisoformat(m["pilot_launch"] + "T00:00:00+00:00")
    ai_labels = ai_label_names(cfg)

    cohort = (
        f"project = {cfg['jira']['project']} AND issuetype != Epic "
        f"AND created >= \"{cfg['jira']['cohort_created_since']}\""
    )
    labeled = within = removed = 0
    violations: list[str] = []
    for key in jira.search_keys(cohort):
        issue = jira.issue(key, ["created", "labels"], expand_changelog=True)
        st = inspect(issue, ai_labels, bot_id, jira.changelog(key, issue.get("changelog")))
        # Collected before the bot-labeled filter: a maintainer hand-applying an
        # ai-triage label to a ticket the bot never touched is the violation
        # most worth seeing.
        if st.human_adds:
            violations.append(f"{key}: {', '.join(st.human_adds)}")
        if not st.bot_first_labeled_at:
            continue
        labeled += 1
        if sla_met(issue["fields"]["created"], st.bot_first_labeled_at, launch):
            within += 1
        if st.opted_out:
            removed += 1
    if not labeled:
        sys.exit("no bot-labeled tickets in the cohort yet - metrics apply to the live phase")

    pct24 = 100 * within / labeled
    removal_rate = removed / labeled
    intro_quoted = ", ".join(f'"{l}"' for l in (m["intro_label"], m["intro_rejected_label"]))
    ai_quoted = ", ".join(f'"{l}"' for l in ai_labels)
    # Scoped to the same pre-registered cohort as the other two metrics, so all
    # three denominators mean the same thing even if an ai-triage label is
    # hand-applied to a ticket outside the window.
    intro = len(jira.search_keys(
        f"{cohort} AND labels in ({intro_quoted}) AND labels in ({ai_quoted})"
    ))

    print(f"tickets labeled    : {labeled}")
    print(f"sorted within 24h  : {pct24:.0f}%  (target >= {m['sorted_within_24h_pct']}%)")
    print(f"label removal rate : {removal_rate:.2f}  (target <= {m['max_label_removal_rate']})")
    print(f"intro outcomes     : {intro}  (target >= {m['min_intro_outcomes']})")
    print(f"convention adds    : {len(violations)}  (non-bot ai-triage label adds)")
    for v in violations:
        print(f"  {v}")

    print(f"\nDECISION: {decide(pct24, removal_rate, intro, m)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
