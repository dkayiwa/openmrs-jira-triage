"""Live-phase weekly metrics plus the pre-registered decision rule.

Everything is computed from Jira changelogs and out/journal.jsonl, so anyone
with read access can reproduce the numbers. Thresholds live in config.toml
[metrics] and must be committed before launch - the design doc's "decision
rule determined before results analyzed". The ADOPT/EXTEND/STOP boundaries
below are a draft to pre-register with the pilot owners.
"""
from __future__ import annotations

import datetime
import json
import os
import sys

from .classifier import LABEL_KEYS
from .run import ROOT, jira_from_env, load_config
from .state import inspect


def main() -> int:
    cfg = load_config()
    jira = jira_from_env(cfg)
    bot_id = os.environ.get("TRIAGE_BOT_ACCOUNT_ID")
    ai_labels = [cfg["labels"][k] for k in LABEL_KEYS]
    m = cfg["metrics"]

    labeled: dict[str, str] = {}
    journal = ROOT / "out" / "journal.jsonl"
    if journal.exists():
        for line in journal.read_text().splitlines():
            row = json.loads(line)
            if row.get("action") == "labeled":
                labeled.setdefault(row["key"], row["at"])
    if not labeled:
        sys.exit("no 'labeled' rows in out/journal.jsonl yet - metrics apply to the live phase")

    within = removed = 0
    for key, at in labeled.items():
        issue = jira.issue(key, ["created", "labels"], expand_changelog=True)
        created = datetime.datetime.fromisoformat(issue["fields"]["created"])
        if datetime.datetime.fromisoformat(at) - created <= datetime.timedelta(hours=24):
            within += 1
        if inspect(issue, ai_labels, bot_id).opted_out:
            removed += 1

    pct24 = 100 * within / len(labeled)
    removal_rate = removed / len(labeled)
    intro_jql = (
        f"project = {cfg['jira']['project']} "
        f"AND labels in ({m['intro_label']}, {m['intro_rejected_label']}) "
        f"AND labels in ({', '.join(ai_labels)})"
    )
    intro = len(jira.search_keys(intro_jql))

    print(f"tickets labeled    : {len(labeled)}")
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
