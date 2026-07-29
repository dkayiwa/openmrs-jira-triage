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

from .classifier import LABEL_KEYS
from .run import jira_from_env, load_config
from .state import inspect


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
    ai_labels = [cfg["labels"][k] for k in LABEL_KEYS]

    cohort_jql = (
        f"project = {cfg['jira']['project']} AND issuetype != Epic "
        f"AND created >= \"{cfg['jira']['cohort_created_since']}\""
    )
    labeled = within = removed = 0
    for key in jira.search_keys(cohort_jql):
        issue = jira.issue(key, ["created", "labels"], expand_changelog=True)
        st = inspect(issue, ai_labels, bot_id)
        if not st.bot_first_labeled_at:
            continue
        labeled += 1
        created = datetime.datetime.fromisoformat(issue["fields"]["created"])
        labeled_at = datetime.datetime.fromisoformat(st.bot_first_labeled_at)
        if labeled_at - max(created, launch) <= datetime.timedelta(hours=24):
            within += 1
        if st.opted_out:
            removed += 1
    if not labeled:
        sys.exit("no bot-labeled tickets in the cohort yet - metrics apply to the live phase")

    pct24 = 100 * within / labeled
    removal_rate = removed / labeled
    intro_quoted = ", ".join(f'"{l}"' for l in (m["intro_label"], m["intro_rejected_label"]))
    ai_quoted = ", ".join(f'"{l}"' for l in ai_labels)
    intro_jql = (
        f"project = {cfg['jira']['project']} "
        f"AND labels in ({intro_quoted}) AND labels in ({ai_quoted})"
    )
    intro = len(jira.search_keys(intro_jql))

    print(f"tickets labeled    : {labeled}")
    print(f"sorted within 24h  : {pct24:.0f}%  (target >= {m['sorted_within_24h_pct']}%)")
    print(f"label removal rate : {removal_rate:.2f}  (target <= {m['max_label_removal_rate']})")
    print(f"intro outcomes     : {intro}  (target >= {m['min_intro_outcomes']})")

    passes = [
        pct24 >= m["sorted_within_24h_pct"],
        removal_rate <= m["max_label_removal_rate"],
        intro >= m["min_intro_outcomes"],
    ]
    if all(passes):
        print("\nDECISION: ADOPT")
    elif removal_rate <= 2 * m["max_label_removal_rate"] and sum(passes) >= 2:
        print("\nDECISION: EXTEND (two weeks)")
    else:
        print("\nDECISION: STOP")
    return 0


if __name__ == "__main__":
    sys.exit(main())
