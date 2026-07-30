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
from .state import PROPERTY_KEY, inspect


def parse_launch(value: str | None) -> datetime.datetime:
    """Midnight UTC on the pre-registered launch date.

    A bare YYYY-MM-DD only: appending a time to anything else yields a malformed
    string and an opaque "Invalid isoformat string" from fromisoformat, which
    says nothing about which config field is wrong.
    """
    if not value:
        sys.exit("set [metrics].pilot_launch in config.toml at launch - it anchors the 24h SLA")
    try:
        date = datetime.date.fromisoformat(value)
    except ValueError:
        sys.exit(f"[metrics].pilot_launch must be a bare date like 2026-08-01, not {value!r}")
    return datetime.datetime.combine(date, datetime.time(), datetime.timezone.utc)


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
    launch = parse_launch(m.get("pilot_launch"))
    ai_labels = ai_label_names(cfg)

    cohort = cfg["jira"]["cohort_jql"].format(since=cfg["jira"]["cohort_created_since"])
    labeled = within = removed = 0
    violations: list[str] = []
    failed: list[str] = []
    replayed: list[str] = []
    cohort_keys = jira.search_keys(cohort)
    for key in cohort_keys:
        # Per-ticket isolation: this walk is hundreds of requests long, and one
        # unreadable ticket must not discard every request before it. A metric
        # computed over an incomplete cohort is worse than no metric, so any
        # failure suppresses the decision below rather than skewing it quietly.
        try:
            issue = jira.issue(key, ["created", "labels"], expand_changelog=True)
            st = inspect(issue, ai_labels, bot_id, jira.changelog(key, issue.get("changelog")))
        except Exception as e:
            failed.append(f"{key}: {type(e).__name__}: {e}"[:200])
            continue
        # Collected before the bot-labeled filter: a maintainer hand-applying an
        # ai-triage label to a ticket the bot never touched is the violation
        # most worth seeing.
        if st.human_adds:
            violations.append(f"{key}: {', '.join(st.human_adds)}")
        if not st.bot_first_labeled_at:
            continue
        # The changelog cannot distinguish a replayed label from a pinned-model
        # one - the bot's credentials wrote both - so the entity property is
        # consulted for the labelled tickets only. Without this the three
        # pre-registered metrics would silently pool two different systems.
        # Read before the accounting below, so a failed read excludes the ticket
        # entirely rather than counting it in the denominator while skipping its
        # contribution to the numerators.
        try:
            prop = jira.get_property(key, PROPERTY_KEY)
        except Exception as e:
            failed.append(f"{key}: reading {PROPERTY_KEY}: {type(e).__name__}: {e}"[:200])
            continue
        labeled += 1
        # A bot-labelled ticket with NO property is unknown provenance, not api
        # provenance, and defaulting it to "api" failed open on precisely the
        # case this withholding exists for. The property is written last, after
        # the label and the comment, so a replayed label whose property write
        # raised is public and unattributed - and would have been counted as
        # pinned-model evidence. evals/run_evals.py already fails closed on an
        # absent source column; this now matches it.
        source = "absent" if prop is None else (prop or {}).get("source", "api")
        if source != "api":
            replayed.append(f"{key}: source={source} "
                            f"classifier={(prop or {}).get('classifier')}")
        prop = prop or {}
        if sla_met(issue["fields"]["created"], st.bot_first_labeled_at, launch):
            within += 1
        if st.opted_out:
            removed += 1
    if failed and not labeled:
        # Checked before the no-tickets exit: a rate-limit or auth window fails
        # every property read at once, and reporting "the bot has labelled
        # nothing" would be the opposite of the truth while discarding the
        # diagnostic list that explains it.
        print(f"{len(failed)} of {len(cohort_keys)} cohort ticket(s) could not be read:")
        for f in failed:
            print(f"  {f}")
        sys.exit("no metrics computed - resolve the failures above and re-run")
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

    # Each line carries its own verdict, at enough precision to show a
    # near-boundary value. Rounding to "95%" against a ">= 95%" target read as a
    # pass while the rule failed it, so the report contradicted its own decision.
    print(f"tickets labeled    : {labeled}")
    for text, ok, target in (
        (f"sorted within 24h  : {pct24:.1f}%", pct24 >= m["sorted_within_24h_pct"],
         f">= {m['sorted_within_24h_pct']}%"),
        (f"label removal rate : {removal_rate:.3f}",
         removal_rate <= m["max_label_removal_rate"],
         f"<= {m['max_label_removal_rate']}"),
        (f"intro outcomes     : {intro}", intro >= m["min_intro_outcomes"],
         f">= {m['min_intro_outcomes']}"),
    ):
        print(f"{text}  [{'PASS' if ok else 'FAIL'}]  (target {target})")
    print(f"convention adds    : {len(violations)}  (non-bot ai-triage label adds)")
    for v in violations:
        print(f"  {v}")

    # Both blockers are reported before returning, so an operator resolving them
    # does not discover the second only after fixing the first.
    if failed:
        print(f"\n{len(failed)} of {len(cohort_keys)} cohort ticket(s) could not be read:")
        for f in failed:
            print(f"  {f}")
    if replayed:
        print(f"\n{len(replayed)} of {labeled} labelled ticket(s) were not labelled by "
              "the pinned model:")
        for r in replayed:
            print(f"  {r}")

    if failed or replayed:
        print("\nNO DECISION.")
        if failed:
            print("- The cohort is incomplete, so the numbers above are a lower bound. "
                  "Re-run once the read failures are resolved.")
        if replayed:
            print("- The pre-registered thresholds assume one pinned model and prompt "
                  "per label, and this cohort mixes two systems. A plain live sweep "
                  "re-classifies the tickets above (the source mismatch alone makes them "
                  "stale; --force is not needed and would re-charge the whole cohort). "
                  "Tickets that have since been opted out or left "
                  f"\"{cfg['jira']['scope_status']}\" cannot be re-classified at all and "
                  "need an explicit recorded decision from the pilot owners.")
            print("- Note the 24h SLA is measured from the FIRST bot label add, so "
                  "re-classifying does not undo a replay run's latency: the timestamp "
                  "does not move when the label is unchanged.")
        return 1
    print(f"\nDECISION: {decide(pct24, removal_rate, intro, m)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
