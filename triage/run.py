"""Triage pilot runner.

Dry-run is the default: it writes proposals, contexts and a journal under
out/ and touches nothing in Jira. --live (gated on credentials and a bot
account id) applies exactly one ai-triage-* label plus one comment per ticket.
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
from .state import PROPERTY_KEY, inspect

ROOT = pathlib.Path(__file__).resolve().parent.parent


def load_config() -> dict:
    with open(ROOT / "config.toml", "rb") as fh:
        return tomllib.load(fh)


def jira_from_env(cfg: dict) -> JiraClient:
    return JiraClient(
        os.environ.get("JIRA_BASE_URL", cfg["jira"]["base_url"]),
        os.environ.get("JIRA_EMAIL"),
        os.environ.get("JIRA_API_TOKEN"),
    )


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


def write_proposals(cfg: dict, out: pathlib.Path, stamp: datetime.datetime, proposals: list) -> pathlib.Path:
    base = out / f"proposals-{stamp:%Y%m%d-%H%M}"
    url = cfg["jira"]["base_url"] + "/browse/"
    with open(base.with_suffix(".csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            ["key", "url", "summary", "proposed_label", "confidence", "rationale",
             "missing_info", "verification_steps", "grade(ok/wrong)", "correct_label", "grader_notes"]
        )
        for issue, c in proposals:
            w.writerow(
                [issue["key"], url + issue["key"], issue["fields"].get("summary", ""), c.label,
                 f"{c.confidence:.2f}", c.rationale, "; ".join(c.missing_info),
                 "; ".join(c.verification_steps), "", "", ""]
            )
    with open(base.with_suffix(".md"), "w") as fh:
        fh.write(
            f"# Triage proposals - {stamp:%Y-%m-%d %H:%M} UTC "
            f"(prompt {cfg['prompt']['version']}, dry-run)\n\n"
            "Grade in the matching CSV: `ok` or `wrong` in grade(ok/wrong); when wrong, set "
            "correct_label to automation_candidate / needs_judgment / needs_more_info.\n\n"
        )
        for issue, c in proposals:
            fh.write(f"## [{issue['key']}]({url}{issue['key']}) {issue['fields'].get('summary', '')}\n\n")
            fh.write(f"**{c.label}** (confidence {c.confidence:.2f})\n\n{c.rationale}\n\n")
            if c.missing_info:
                fh.write("Missing info: " + "; ".join(c.missing_info) + "\n\n")
            if c.verification_steps:
                fh.write("Verification: " + "; ".join(c.verification_steps) + "\n\n")
    return base


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="OpenMRS O3 AI triage pilot")
    ap.add_argument("--live", action="store_true", help="apply labels/comments (default: dry-run)")
    ap.add_argument("--limit", type=int, default=0, help="max tickets this run (0 = all)")
    ap.add_argument("--keys", help="comma-separated issue keys (skips the JQL sweep)")
    ap.add_argument("--no-classify", action="store_true", help="fetch and assemble contexts only")
    ap.add_argument("--force", action="store_true", help="dry-run: reclassify already-labeled tickets")
    args = ap.parse_args(argv)

    cfg = load_config()
    jira = jira_from_env(cfg)
    bot_id = os.environ.get("TRIAGE_BOT_ACCOUNT_ID")
    if args.live and not (jira.authenticated and bot_id):
        sys.exit("--live needs JIRA_EMAIL, JIRA_API_TOKEN and TRIAGE_BOT_ACCOUNT_ID")

    out = ROOT / "out"
    (out / "contexts").mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now(datetime.timezone.utc)

    ac_field = ctx.discover_ac_field(jira, cfg["jira"].get("acceptance_criteria_field", ""))
    if not ac_field:
        print("WARN: Acceptance Criteria field not found; classifying without it", file=sys.stderr)

    ai_labels = [cfg["labels"][k] for k in LABEL_KEYS]
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
        issue = jira.issue(key, fields, expand_changelog=True)
        st = inspect(issue, ai_labels, bot_id)
        text = ctx.assemble(jira, issue, ac_field, blocked)
        chash = ctx.content_hash(text)
        (out / "contexts" / f"{key}.txt").write_text(text)

        row: dict = {
            "at": stamp.isoformat(), "key": key, "hash": chash,
            "prompt": cfg["prompt"]["version"], "mode": "live" if args.live else "dry-run",
        }
        unchanged = (
            not args.live
            or (jira.get_property(key, PROPERTY_KEY) or {}).get("contentHash") == chash
        )
        if st.opted_out:
            row["action"] = "skip-opted-out"
            row["by"] = st.opted_out_by
        elif st.ai_labels_present and unchanged and not args.force:
            row["action"] = "skip-already-triaged"
            row["labels"] = st.ai_labels_present
        elif classifier is None:
            row["action"] = "context-only"
        else:
            c = classifier.classify(text)
            if c.refused:
                row["action"] = "error-refusal"
            else:
                row.update(action="proposed", label=c.label, confidence=c.confidence, model=c.model)
                proposals.append((issue, c))
                if args.live:
                    label = cfg["labels"][c.label]
                    jira.update_labels(key, [label], [l for l in st.ai_labels_present if l != label])
                    jira.add_comment(key, comment_body(cfg, c))
                    jira.set_property(key, PROPERTY_KEY, {
                        "contentHash": chash, "label": label,
                        "prompt": cfg["prompt"]["version"], "at": row["at"],
                    })
                    row["action"] = "labeled"
        if st.human_adds:
            row["convention_violation_adds"] = st.human_adds
        with open(journal, "a") as fh:
            fh.write(json.dumps(row) + "\n")
        suffix = f" -> {row['label']}" if row.get("label") else ""
        print(f"  {key}: {row['action']}{suffix}")

    if proposals:
        base = write_proposals(cfg, out, stamp, proposals)
        print(f"\nGrading sheet: {base}.csv (and .md); contexts in {out / 'contexts'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
