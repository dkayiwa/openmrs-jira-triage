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


def parse_launch(value: str | None, today: datetime.date | None = None) -> datetime.datetime:
    """Midnight UTC on the pre-registered launch date.

    A bare YYYY-MM-DD only: appending a time to anything else yields a malformed
    string and an opaque "Invalid isoformat string" from fromisoformat, which
    says nothing about which config field is wrong.

    Only an obviously-mistyped year is refused here; the precise check lives in
    sla_met, where the data is. A first attempt rejected any future date and
    the test suite immediately objected: the fixtures launch tomorrow, and so
    does a real pilot whose config is committed the day before go-live. A gate
    that turns red on the eve of launch is a worse bug than the one it guards.
    """
    if not value:
        sys.exit("set [metrics].pilot_launch in config.toml at launch - it anchors the 24h SLA")
    try:
        date = datetime.date.fromisoformat(value)
    except ValueError:
        sys.exit(f"[metrics].pilot_launch must be a bare date like 2026-08-01, not {value!r}")
    today = today or datetime.datetime.now(datetime.timezone.utc).date()
    if (date - today).days > 90:
        sys.exit(f"[metrics].pilot_launch is {value}, over 90 days after today ({today}). "
                 "That is a typed year rather than a launch date, and it would make every "
                 "ticket report as sorted within 24h - see sla_met.")
    return datetime.datetime.combine(date, datetime.time(), datetime.timezone.utc)


def one_line(value, limit: int = 200) -> str:
    """Collapse an externally-sourced string to a single bounded line.

    This report IS the pilot's decision artifact: it ends in a DECISION line
    that Dennis and Veronica read weekly, and every list below is printed one
    item per line. Any value that reaches it carrying a newline can forge a
    line - including a convincing "DECISION: ADOPT" above the genuine verdict.

    run.py already defends the other end of this pipe. FileClassifier flattens
    the classifier name a classifications file declares, and says why in as
    many words: "an unchecked newline could forge a per-ticket log line or a
    'DECISION: ADOPT' line in the pilot's own decision artifact". That guard
    only covers what this pipeline writes. These values are read back out of
    Jira entity properties, out of Jira display names, and out of exception
    text quoting an API response body - none of which this code wrote, and the
    property store in particular is editable by anyone with API access to the
    issue. Defending the write side alone was the gap.
    """
    return " ".join(str(value).split())[:limit]


def sla_met(created: str, labeled_at: str, launch: datetime.datetime) -> bool:
    """Was the ticket sorted within 24h of entering scope?

    Measured from max(created, launch) because the initial backlog cohort is
    older than 24h by definition.

    A label that predates the launch is refused rather than measured. Because
    the start is max(created, launch), a launch after the label makes the
    elapsed time NEGATIVE, and negative is comfortably within 24 hours - so
    every such ticket reports as sorted on time. Measured before this guard: a
    ticket created 2026-08-10 and labelled 2026-09-20, forty-one days late,
    returned True under a launch of 2027-08-01. The report then prints
    "sorted within 24h : 100.0%  [PASS]" and a pre-registered decision can
    reach ADOPT on a number nobody measured.

    It is reachable from one hand-typed string: pilot_launch is set once, at
    launch, and a mistyped year is the ordinary way that goes wrong. Raising
    puts the ticket in metrics.py's `failed` list, which suppresses the
    decision entirely - the right answer, since the anchor is wrong for the
    whole cohort and not just this ticket.
    """
    start = max(datetime.datetime.fromisoformat(created), launch)
    labeled = datetime.datetime.fromisoformat(labeled_at)
    if labeled < launch:
        raise ValueError(
            f"labelled {labeled.date()} but [metrics].pilot_launch is "
            f"{launch.date()}: the bot cannot have labelled a ticket before the "
            "pilot began, so the launch date is wrong")
    return labeled - start <= datetime.timedelta(hours=24)


def validate_thresholds(m: dict) -> list[str]:
    """Unit errors in the pre-registered thresholds; empty if they are sane.

    [metrics] puts a percentage and a fraction next to each other:

        sorted_within_24h_pct  = 95      # out of 100
        max_label_removal_rate = 0.10    # out of 1

    so writing 10 in the second field, meaning "10%", is the natural mistake -
    and it silently disables the kill metric. Measured: a cohort with 26% of
    its labels removed, which is nearly triple the real threshold, decides
    ADOPT instead of STOP. The removal rate is the one metric that can stop the
    pilot, the comparison is `<=`, and every rate is <= 10, so it can never
    fail again. Nothing in the report looks wrong; it prints the threshold it
    was given.

    These are pre-registered and set once, so they are never re-read with fresh
    eyes after launch. That is exactly the kind of value worth range-checking
    rather than trusting.
    """
    errors = []
    pct = m.get("sorted_within_24h_pct")
    if not isinstance(pct, (int, float)) or isinstance(pct, bool) or not 1 < pct <= 100:
        errors.append(f"sorted_within_24h_pct is {pct!r}; it is a percentage out of 100 "
                      "(95 means 95%), so a value at or below 1 is a fraction in the "
                      "wrong field")
    rate = m.get("max_label_removal_rate")
    if not isinstance(rate, (int, float)) or isinstance(rate, bool) or not 0 < rate <= 1:
        errors.append(f"max_label_removal_rate is {rate!r}; it is a fraction of 1 "
                      "(0.10 means 10%), so a value above 1 disables the kill metric "
                      "entirely - every removal rate would pass")
    intro = m.get("min_intro_outcomes")
    if not isinstance(intro, int) or isinstance(intro, bool) or intro < 0:
        errors.append(f"min_intro_outcomes is {intro!r}; it counts tickets, so it must "
                      "be a non-negative whole number")
    return errors


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
    # Before anything is counted: a threshold in the wrong unit does not make
    # the numbers wrong, it makes the verdict wrong, which is harder to notice.
    threshold_errors = validate_thresholds(m)
    if threshold_errors:
        sys.exit("[metrics] thresholds are not usable:\n  " + "\n  ".join(threshold_errors))
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
            failed.append(one_line(f"{key}: {type(e).__name__}: {e}"))
            continue
        # Collected before the bot-labeled filter: a maintainer hand-applying an
        # ai-triage label to a ticket the bot never touched is the violation
        # most worth seeing.
        if st.human_adds:
            violations.append(one_line(f"{key}: {', '.join(st.human_adds)}"))
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
            failed.append(one_line(f"{key}: reading {PROPERTY_KEY}: {type(e).__name__}: {e}"))
            continue
        # Computed before anything is counted. A missing `created`, or a
        # timestamp fromisoformat cannot parse from either the field or the
        # changelog, would otherwise raise out of the loop and discard every
        # ticket already walked along with the `failed` diagnostics explaining
        # why - the opposite of the per-ticket isolation this walk is built on.
        try:
            met_sla = sla_met(issue["fields"]["created"], st.bot_first_labeled_at, launch)
        except Exception as e:
            failed.append(one_line(f"{key}: computing the 24h SLA: {type(e).__name__}: {e}"))
            continue
        labeled += 1
        # A bot-labelled ticket with NO property is unknown provenance, not api
        # provenance, and defaulting it to "api" failed open on precisely the
        # case this withholding exists for. The property is written last, after
        # the label and the comment, so a replayed label whose property write
        # raised is public and unattributed - and would have been counted as
        # pinned-model evidence. evals/run_evals.py already fails closed on an
        # absent source column; this now matches it.
        # An empty property is as unattributed as a missing one, and a property
        # that is not a dict at all (hand-set, or written by something else) must
        # be recorded rather than raising AttributeError and killing the whole
        # run - per-ticket isolation is the point of `failed`.
        if not prop:
            source, classifier = "absent", None
        elif isinstance(prop, dict):
            source, classifier = prop.get("source", "api"), prop.get("classifier")
        else:
            source, classifier = f"malformed ({type(prop).__name__})", None
        if source != "api":
            replayed.append(f"{key}: source={one_line(source, 60)} classifier={one_line(classifier, 120)}")
        if met_sla:
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
