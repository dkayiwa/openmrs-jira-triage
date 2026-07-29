"""Pre-launch checks from the pilot's Jira-setup section.

Verifies connectivity/auth, that the scope JQL runs (including the
development[] clause), that the Acceptance Criteria field and "To Do" status
exist, and - with --scratch and bot credentials - that Jira really rejects '/'
in labels, which is why config.toml uses hyphenated ai-triage-* names instead
of the design doc's ai-triage/* form.
"""
from __future__ import annotations

import argparse
import os
import sys

from . import context as ctx
from .jira import JiraError
from .run import jira_from_env, load_config


def check(label: str, ok: bool, detail: str = "") -> bool:
    print(f"[{'PASS' if ok else 'FAIL'}] {label}" + (f" - {detail}" if detail else ""))
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="triage pilot preflight")
    ap.add_argument("--scratch", help="issue key for the label-charset write test (needs bot auth)")
    args = ap.parse_args(argv)

    cfg = load_config()
    jira = jira_from_env(cfg)
    ok = True

    info = jira.server_info()
    ok &= check("jira reachable", True, f"{info.get('baseUrl')} ({info.get('deploymentType')})")
    me = jira.myself()
    print(f"       auth: {'as ' + (me.get('displayName') or '?') if me else 'anonymous (read-only)'}")

    statuses: set[str] = set()
    for itype in jira.project_statuses(cfg["jira"]["project"]):
        statuses.update(s["name"] for s in itype["statuses"])
    ok &= check('"To Do" status exists', "To Do" in statuses, f"project statuses: {sorted(statuses)}")

    ac = ctx.discover_ac_field(jira, cfg["jira"].get("acceptance_criteria_field", ""))
    ok &= check("Acceptance Criteria field", bool(ac), ac or "not found")

    jql = cfg["jira"]["scope_jql"].format(since=cfg["jira"]["cohort_created_since"])
    dev_clause = " AND development[pullrequests].all = 0"
    try:
        n = len(jira.search_keys(jql))
        ok &= check("scope JQL (with development[] clause)", True, f"{n} ticket(s); doc estimates ~35")
    except JiraError as e:
        check("scope JQL (with development[] clause)", False, str(e)[:200])
        ok = False
        try:
            n = len(jira.search_keys(jql.replace(dev_clause, "")))
            print(f"       without the clause: {n} ticket(s) - the dev-panel JQL may need auth "
                  "or the GitHub-for-Jira app")
        except JiraError as e2:
            print(f"       fallback sweep also failed: {str(e2)[:200]}")

    if args.scratch and jira.authenticated:
        slash_label = "ai-triage/charset-test"
        try:
            jira.update_labels(args.scratch, [slash_label], [])
            jira.update_labels(args.scratch, [], [slash_label])
            check("slash rejected in labels", False,
                  f"Jira accepted {slash_label!r}; the doc's ai-triage/* names would work after all")
            ok = False
        except JiraError:
            check("slash rejected in labels", True, "hyphenated ai-triage-* names are required")
        hyphen_label = "ai-triage-charset-test"
        try:
            jira.update_labels(args.scratch, [hyphen_label], [])
            jira.update_labels(args.scratch, [], [hyphen_label])
            ok &= check("hyphenated label accepted", True, hyphen_label)
        except JiraError as e:
            ok &= check("hyphenated label accepted", False, str(e)[:200])
    else:
        print("       (label-charset write test skipped: pass --scratch O3-XXXX with bot credentials)")

    key_state = "ANTHROPIC_API_KEY set" if os.environ.get("ANTHROPIC_API_KEY") \
        else "no env key (SDK will use an `ant auth login` profile if present)"
    print(f"       Anthropic credentials: {key_state}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
