"""Triage pilot runner.

Dry-run is the default: it writes proposals, contexts and a journal under
out/ and touches nothing in Jira. --live (gated on credentials, and on the
configured bot account id actually matching those credentials) applies exactly
one ai-triage-* label plus one comment per ticket.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import json
import os
import pathlib
import sys
import tomllib

from . import context as ctx
from .classifier import LABEL_KEYS, Classifier
from .jira import JiraClient
from .state import PROPERTY_KEY, TicketState, inspect

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _load_dotenv(env: pathlib.Path | None = None) -> None:
    """Load .env into os.environ (existing env vars win)."""
    env = env or ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def load_config() -> dict:
    _load_dotenv()
    with open(ROOT / "config.toml", "rb") as fh:
        return tomllib.load(fh)


def ai_label_names(cfg: dict) -> list[str]:
    """The pilot's ai-triage-* label names.

    Single-sourced because run.py applies these and metrics.py measures them:
    a divergence between the two would silently measure the wrong labels.
    """
    return [cfg["labels"][k] for k in LABEL_KEYS]


def jira_from_env(cfg: dict) -> JiraClient:
    return JiraClient(
        os.environ.get("JIRA_BASE_URL", cfg["jira"]["base_url"]),
        os.environ.get("JIRA_EMAIL"),
        os.environ.get("JIRA_API_TOKEN"),
    )


def bot_identity_error(jira: JiraClient, bot_id: str | None) -> str | None:
    """Describe a TRIAGE_BOT_ACCOUNT_ID that cannot belong to these credentials.

    A wrong bot id fails silently and badly: the bot's own label flips read as
    human removals (permanently opting those tickets out of the pilot), its own
    adds are logged as convention violations, and its own comments leak into
    classifier contexts. Returns None when unauthenticated (nothing to compare).
    """
    if not (bot_id and jira.authenticated):
        return None
    me = jira.myself()
    if not me:
        return None
    actual = me.get("accountId")
    if actual and actual != bot_id:
        return (f"TRIAGE_BOT_ACCOUNT_ID is {bot_id} but these credentials are "
                f"{me.get('displayName') or '?'} ({actual})")
    return None


def plan_ticket(st: TicketState, unchanged: bool, force: bool,
                can_classify: bool) -> str | None:
    """The skip action for this ticket, or None to classify it.

    Opt-out is tested first and unconditionally: --force must never re-label a
    ticket a human removed an ai-triage label from.
    """
    if st.opted_out:
        return "skip-opted-out"
    if st.ai_labels_present and unchanged and not force:
        return "skip-already-triaged"
    if not can_classify:
        return "context-only"
    return None


def plan_label_writes(present: list[str], label: str) -> tuple[list[str], list[str], bool]:
    """(labels_to_add, labels_to_remove, post_comment).

    Comment only when the label is new for the ticket: re-runs after content
    edits or a lost property must never post duplicate comments to watchers.

    The trade this makes: a crash between the label write and the comment write
    leaves the ticket labelled with no comment, and no later run will add one
    (the label is already there). The journal's error row is the recovery
    signal. A missing comment is quieter than a duplicate one to every watcher.
    """
    is_new = label not in present
    stale = [l for l in present if l != label]
    return ([label] if is_new else [], stale, is_new)


def comment_body(cfg: dict, c) -> str:
    label = cfg["labels"][c.label]
    lines = [f"AI triage: {{{{{label}}}}}", "", c.rationale]
    if c.label == "needs_more_info" and c.missing_info:
        lines += ["", "Missing information:"] + [f"- {m}" for m in c.missing_info]
    if c.label == "automation_candidate" and c.verification_steps:
        lines += ["", "How to verify:"] + [f"- {v}" for v in c.verification_steps]
    lines += [
        "",
        "_Applied by the triage pilot bot from this ticket's visible content only. "
        f"Removing the label opts the ticket out of the pilot. (prompt {cfg['prompt']['version']})_",
    ]
    return "\n".join(lines)


def write_proposals(cfg: dict, out: pathlib.Path, stamp: datetime.datetime, proposals: list,
                    live: bool) -> pathlib.Path:
    # Seconds included: two quick --keys runs in the same minute would
    # otherwise overwrite a grading sheet someone had already started on.
    base = out / f"proposals-{stamp:%Y%m%d-%H%M%S}"
    url = cfg["jira"]["base_url"] + "/browse/"
    with open(base.with_suffix(".csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        # content_hash pins each grade to the exact context it was made against,
        # so importing into the eval set can detect a context that has since
        # been overwritten by a later dry-run.
        w.writerow(
            ["key", "url", "summary", "proposed_label", "confidence", "rationale",
             "missing_info", "verification_steps", "content_hash",
             "grade(ok/wrong)", "correct_label", "grader_notes"]
        )
        for issue, c, chash in proposals:
            w.writerow(
                [issue["key"], url + issue["key"], issue["fields"].get("summary", ""), c.label,
                 f"{c.confidence:.2f}", c.rationale, "; ".join(c.missing_info),
                 "; ".join(c.verification_steps), chash, "", "", ""]
            )
    with open(base.with_suffix(".md"), "w") as fh:
        # The mode is stated because this file is also written on live runs,
        # where it is an audit trail of writes that already happened - not a
        # preview of writes that did not.
        fh.write(
            f"# Triage proposals - {stamp:%Y-%m-%d %H:%M} UTC "
            f"(prompt {cfg['prompt']['version']}, {'LIVE - labels and comments applied' if live else 'dry-run'})\n\n"
        )
        if not live:
            fh.write(
                "Grade in the matching CSV: `ok` or `wrong` in grade(ok/wrong); when wrong, set "
                "correct_label to automation_candidate / needs_judgment / needs_more_info.\n\n"
            )
        for issue, c, _ in proposals:
            fh.write(f"## [{issue['key']}]({url}{issue['key']}) {issue['fields'].get('summary', '')}\n\n")
            fh.write(f"**{c.label}** (confidence {c.confidence:.2f})\n\n{c.rationale}\n\n")
            if c.missing_info:
                fh.write("Missing info: " + "; ".join(c.missing_info) + "\n\n")
            if c.verification_steps:
                fh.write("Verification: " + "; ".join(c.verification_steps) + "\n\n")
    return base


def main(argv=None, out: pathlib.Path | None = None) -> int:
    ap = argparse.ArgumentParser(description="OpenMRS O3 AI triage pilot")
    ap.add_argument("--live", action="store_true", help="apply labels/comments (default: dry-run)")
    ap.add_argument("--limit", type=int, default=0, help="max tickets this run (0 = all)")
    ap.add_argument("--keys", help="comma-separated issue keys (skips the JQL sweep)")
    ap.add_argument("--no-classify", action="store_true", help="fetch and assemble contexts only")
    ap.add_argument("--force", action="store_true",
                    help="reclassify already-triaged tickets (opt-outs are still respected)")
    args = ap.parse_args(argv)

    cfg = load_config()
    jira = jira_from_env(cfg)
    bot_id = os.environ.get("TRIAGE_BOT_ACCOUNT_ID")
    if args.live and not (jira.authenticated and bot_id):
        sys.exit("--live needs JIRA_EMAIL, JIRA_API_TOKEN and TRIAGE_BOT_ACCOUNT_ID")
    mismatch = bot_identity_error(jira, bot_id)
    if mismatch and args.live:
        sys.exit(mismatch)
    if mismatch:
        print(f"WARN: {mismatch}", file=sys.stderr)
    # Gated on pilot_launch so this stays quiet during pre-launch grading, when
    # there are no bot comments or ai-triage labels for it to be about; once the
    # pilot is live it fires before any classification is paid for.
    if not bot_id and cfg["metrics"].get("pilot_launch"):
        print("WARN: TRIAGE_BOT_ACCOUNT_ID unset; the bot's own comments will not be "
              "filtered out of contexts and every ai-triage removal counts as an opt-out",
              file=sys.stderr)

    out = out or ROOT / "out"
    (out / "contexts").mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now(datetime.timezone.utc)

    ac_field = ctx.discover_ac_field(jira, cfg["jira"].get("acceptance_criteria_field", ""))
    if not ac_field:
        print("WARN: Acceptance Criteria field not found; classifying without it", file=sys.stderr)

    ai_labels = ai_label_names(cfg)
    blocked = list(cfg["bots"]["blocked_account_ids"]) + ([bot_id] if bot_id else [])

    if args.keys:
        keys = [k.strip() for k in args.keys.split(",") if k.strip()]
    else:
        jql = cfg["jira"]["scope_jql"].format(since=cfg["jira"]["cohort_created_since"])
        try:
            keys = jira.search_keys(jql)
        except Exception as e:
            dev_clause = " AND development[pullrequests].all = 0"
            if dev_clause in jql and "development" in str(e).lower():
                print("WARN: development[] JQL clause rejected here; sweeping without the "
                      "no-linked-PRs filter", file=sys.stderr)
                keys = jira.search_keys(jql.replace(dev_clause, ""))
            else:
                raise
    if args.limit:
        keys = keys[: args.limit]
    print(f"{len(keys)} ticket(s) in scope")

    classifier = None
    if not args.no_classify:
        classifier = Classifier(
            cfg["claude"]["model"], cfg["claude"]["max_tokens"],
            (ROOT / "prompt" / "system.md").read_text(),
        )

    fields = ctx.ISSUE_FIELDS + ([ac_field] if ac_field else [])
    proposals: list = []
    journal = out / "journal.jsonl"
    for key in keys:
        row: dict = {
            "key": key, "prompt": cfg["prompt"]["version"],
            "mode": "live" if args.live else "dry-run",
        }
        st = None
        # Per-ticket isolation: one bad ticket (API hiccup, truncated JSON,
        # Jira write failure) must not abort the rest of the sweep.
        try:
            issue = jira.issue(key, fields, expand_changelog=True)
            st = inspect(issue, ai_labels, bot_id,
                         jira.changelog(key, issue.get("changelog")))
            text = ctx.assemble(jira, issue, ac_field, blocked)
            chash = ctx.content_hash(text)
            (out / "contexts" / f"{key}.txt").write_text(text)
            row["hash"] = chash

            # Dry-run skips the property read: nothing was written, so the
            # stored hash reflects the last *live* run, and re-grading the same
            # unchanged ticket would just re-spend on classification.
            #
            # The label is a function of the content AND the prompt, so both are
            # compared: a prompt bump must re-sweep the cohort rather than leave
            # it graded under two prompt versions. Re-classification is quiet
            # unless the label actually flips.
            prop = {} if not args.live else (jira.get_property(key, PROPERTY_KEY) or {})
            unchanged = (
                not args.live
                or (prop.get("contentHash") == chash
                    and prop.get("prompt") == cfg["prompt"]["version"])
            )
            action = plan_ticket(st, unchanged, args.force, classifier is not None)
            if action:
                row["action"] = action
                if action == "skip-opted-out":
                    row["by"] = st.opted_out_by
                elif action == "skip-already-triaged":
                    row["labels"] = st.ai_labels_present
            else:
                c = classifier.classify(text)
                if c.refused:
                    row["action"] = "error-refusal"
                else:
                    row.update(action="proposed", label=c.label, confidence=c.confidence, model=c.model)
                    proposals.append((issue, c, chash))
                    if args.live:
                        label = cfg["labels"][c.label]
                        add, remove, post_comment = plan_label_writes(st.ai_labels_present, label)
                        if add or remove:
                            jira.update_labels(key, add, remove)
                        if post_comment:
                            jira.add_comment(key, comment_body(cfg, c))
                        jira.set_property(key, PROPERTY_KEY, {
                            "contentHash": chash, "label": label,
                            "prompt": cfg["prompt"]["version"],
                            "at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                        })
                        row["action"] = "labeled" if post_comment else "refreshed"
        except Exception as e:
            row["action"] = "error"
            row["error"] = f"{type(e).__name__}: {e}"[:300]
        if st and st.human_adds:
            row["convention_violation_adds"] = st.human_adds
        row["at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        with open(journal, "a") as fh:
            fh.write(json.dumps(row) + "\n")
        suffix = f" -> {row['label']}" if row.get("label") else ""
        if row["action"] == "error":
            suffix = f" ({row['error']})"
        print(f"  {key}: {row['action']}{suffix}")

    if proposals:
        base = write_proposals(cfg, out, stamp, proposals, args.live)
        print(f"\nGrading sheet: {base}.csv (and .md); contexts in {out / 'contexts'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
