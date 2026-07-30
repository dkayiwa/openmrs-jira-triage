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
from .classifier import (
    LABEL_KEYS,
    MAX_ITEM,
    MAX_ITEMS,
    MAX_RATIONALE,
    SCHEMA,
    Classification,
    Classifier,
    validate_classification,
)
from .github import GitHubClient
from .jira import JiraClient
from .state import PROPERTY_KEY, TicketState, inspect

ROOT = pathlib.Path(__file__).resolve().parent.parent

# A systemic fault (missing permission, rotated key, API outage) fails every
# ticket. Each failure re-spends a paid classification on the next sweep, so the
# sweep stops rather than working through the whole cohort six times a day.
CONSECUTIVE_ERROR_LIMIT = 5

# The grading sheet's columns. evals/run_evals.py reads several of these by
# name, so they are named once here; a round-trip test pins the two together.
# content_hash pins each grade to the exact context it was made against, so the
# import can detect a context overwritten by a later dry-run.
PROPOSAL_COLUMNS = [
    "key", "url", "summary", "proposed_label", "confidence", "rationale",
    "missing_info", "verification_steps", "content_hash", "source",
    "grade(ok/wrong)", "correct_label", "grader_notes",
]


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


def classification_entry_schema() -> dict:
    """The schema of one entry in a classifications file.

    Derived from the response SCHEMA rather than restated, so the two cannot
    drift, plus the two things the response schema cannot carry: content_hash
    (which binds the entry to a context) and the magnitude caps.
    """
    props = {k: dict(v) for k, v in SCHEMA["properties"].items()}
    props["rationale"]["maxLength"] = MAX_RATIONALE
    for key in ("missing_info", "verification_steps"):
        props[key]["maxItems"] = MAX_ITEMS
        props[key]["items"] = dict(props[key]["items"], maxLength=MAX_ITEM)
    props["confidence"].update(minimum=0, maximum=1)
    props["content_hash"] = {
        "type": "string",
        "description": "the content_hash of the context this classifies, from tickets[]",
    }
    return {
        "type": "object",
        "properties": props,
        "required": [*SCHEMA["required"], "content_hash"],
        "additionalProperties": False,
    }


class NotClassified(Exception):
    """This ticket is not in the classifications file.

    A routine, free, expected condition - a file legitimately covers a subset of
    the sweep. Distinct from an error so it never reaches the consecutive-error
    circuit breaker, which exists to cap *paid* API faults: counting misses
    there aborted the sweep partway and dropped classifications that were
    perfectly good.
    """


class FileClassifier:
    """Serves classifications produced outside the pipeline.

    Same interface as Classifier, so a run needs no Anthropic credential: an
    agent in a Claude Code session (or anything else) reads out/contexts/*.txt
    and writes a classifications file, and this replays it.

    Lookup is by content hash rather than ticket key, which makes staleness
    detection free: if the ticket was edited between being classified and being
    applied, the freshly assembled context hashes differently and no
    classification is found, so the stale label is never written.
    """

    def __init__(self, path: pathlib.Path, prompt_version: str):
        # Every failure here is fatal and named, before any ticket is fetched:
        # this file decides what gets written to public tickets.
        try:
            doc = json.loads(path.read_text())
        except FileNotFoundError:
            sys.exit(f"{path}: no such file")
        except json.JSONDecodeError as e:
            sys.exit(f"{path}: not valid JSON ({e})")
        if not isinstance(doc, dict):
            sys.exit(f"{path}: expected a JSON object, got {type(doc).__name__}")
        if doc.get("prompt_version") != prompt_version:
            sys.exit(f"{path}: classified against prompt {doc.get('prompt_version')!r} "
                     f"but config.toml pins {prompt_version!r}")
        # The one header field, and the only part of this document that
        # validate_classification never sees. It reaches the run log, the entity
        # property and the weekly metrics report verbatim, so an unchecked
        # newline could forge a per-ticket log line or a "DECISION: ADOPT" line
        # in the pilot's own decision artifact.
        declared = doc.get("classifier") or "unattributed"
        if not isinstance(declared, str):
            sys.exit(f"{path}: classifier must be a string, got {type(declared).__name__}")
        self.model = " ".join(declared.split())[:120] or "unattributed"

        entries = doc.get("classifications") or {}
        if not isinstance(entries, dict):
            sys.exit(f"{path}: classifications must be an object, "
                     f"got {type(entries).__name__}")
        if not entries:
            sys.exit(f"{path}: no classifications in the file")
        self.by_hash: dict[str, Classification] = {}
        self.keys: dict[str, str] = {}
        owner: dict[str, str] = {}
        for key, entry in entries.items():
            if not isinstance(entry, dict):
                sys.exit(f"{path}: {key} must be an object, got {type(entry).__name__}")
            entry = dict(entry)
            chash = entry.pop("content_hash", None)
            if not chash:
                sys.exit(f"{path}: {key} has no content_hash, so it cannot be matched "
                         "to a context safely")
            if not isinstance(chash, str):
                sys.exit(f"{path}: {key} content_hash must be a string, "
                         f"got {type(chash).__name__}")
            if chash in owner:
                # Lookup is by hash, so a repeat would silently overwrite - and
                # if the two disagree, which label lands would depend on dict
                # order rather than on anything the author intended.
                sys.exit(f"{path}: {key} and {owner[chash]} share content_hash {chash}")
            owner[chash] = key
            errors = validate_classification(entry)
            if errors:
                sys.exit(f"{path}: {key} is not a valid classification: {'; '.join(errors)}")
            self.keys[chash] = key
            self.by_hash[chash] = Classification(
                label=entry["label"], rationale=entry["rationale"],
                missing_info=list(entry["missing_info"]),
                verification_steps=list(entry["verification_steps"]),
                confidence=float(entry["confidence"]), model=self.model,
            )

    def classify(self, context: str) -> Classification:
        chash = ctx.content_hash(context)
        found = self.by_hash.get(chash)
        if found is None:
            raise NotClassified(
                f"no classification for content hash {chash}; not in this batch, "
                "or the ticket changed since it was classified"
            )
        # The hash guards against edited content but not against *mispaired*
        # content: swapping two entries' content_hash values would apply each
        # ticket's label and rationale to the other, silently. Every context
        # starts with "TICKET: <key>", so the entry's own key is checkable.
        declared = self.keys[chash]
        actual = context.splitlines()[0].removeprefix("TICKET:").strip()
        # Case-insensitive: Jira accepts and normalises a lowercase key, so a
        # gather run using --keys o3-50 writes the manifest under "o3-50" while
        # the context says "O3-50". Rejecting that would refuse a file built
        # faithfully from the manifest.
        if actual.upper() != declared.upper():
            raise RuntimeError(
                f"classification is filed under {declared} but this content is "
                f"{actual}; the content_hash values are mispaired"
            )
        return found


def bot_identity_error(jira: JiraClient, bot_id: str | None) -> str | None:
    """Describe a TRIAGE_BOT_ACCOUNT_ID that cannot belong to these credentials.

    A wrong bot id fails silently and badly: the bot's own label flips read as
    human removals (permanently opting those tickets out of the pilot), its own
    adds are logged as convention violations, and its own comments leak into
    classifier contexts. Returns None when unauthenticated (nothing to compare).
    """
    if not (bot_id and jira.authenticated):
        return None
    # Fails closed: an unusable /myself means the check could not run, which is
    # not the same as passing. Proceeding unverified is how the silent
    # cohort-wide opt-out this guard exists to prevent would happen anyway.
    me = jira.myself()
    actual = (me or {}).get("accountId")
    if not actual:
        return "could not verify TRIAGE_BOT_ACCOUNT_ID: /myself returned no account"
    if actual != bot_id:
        return (f"TRIAGE_BOT_ACCOUNT_ID is {bot_id} but these credentials are "
                f"{me.get('displayName') or '?'} ({actual})")
    return None


def plan_ticket(st: TicketState, unchanged: bool, force: bool, can_classify: bool,
                out_of_scope: bool = False, has_open_pr: bool = False) -> str | None:
    """The skip action for this ticket, or None to classify it.

    Opt-out, out-of-scope and open-PR are tested first and unconditionally:
    --force is about reclassifying already-triaged tickets, never about
    re-labelling one a human opted out of, or one that is not the pilot's to
    sort - because it has left scope, or because it is already in review.

    An open PR outranks "already-triaged" on purpose. Both would leave an
    unchanged ticket alone, but only this one keeps it out of the manifest, so a
    ticket that gained a PR after being labelled stops being offered for
    re-classification on the next prompt bump.
    """
    if st.opted_out:
        return "skip-opted-out"
    if out_of_scope:
        return "skip-out-of-scope"
    if has_open_pr:
        return "skip-open-pr"
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


def csv_safe(text: str) -> str:
    """Defuse spreadsheet formula injection in a grading-sheet cell.

    The sheet is both the human deliverable and the eval harness's input, and
    Sheets/Excel evaluate a leading =, +, - or @ when the file is opened - so an
    =IMPORTXML(...) rationale would exfiltrate the row to a third party with no
    click at all. A leading apostrophe forces the cell to text.
    """
    text = str(text or "")
    return "'" + text if text[:1] in ("=", "+", "-", "@", "\t", "\r", "\n") else text


def wiki_safe(text: str) -> str:
    r"""Neutralise Jira wiki constructs in model output that act on third parties.

    Ticket text is untrusted and reaches the model, so the model's output is
    untrusted too. In a v2 comment body `[~accountid:...]` renders as a real
    @mention that notifies that account, and `!url!` embeds a remote image
    fetched by everyone who views the ticket - both actions on people who never
    asked, re-fired on every content change. Cohort tickets already contain
    literal accountid tokens, so this is reachable, not theoretical.

    Escaping `[` and `!` disables mentions, links and images. The safety does
    not depend on how Jira renders the escape: either way the construct no
    longer fires.

    Backslashes are removed FIRST, for two reasons. Escaping by prefix is not
    idempotent: text supplying its own `\` turns our `\[` into `\\[`, and Jira
    consumes `\\` as a forced line break, leaving the `[` live - so the mention
    fired anyway (verified against rendered Jira output). And `\\` is itself a
    line break, so it could forge a second `AI triage: {{...}}` line and a fake
    footer, making the comment state a label that was never applied. Collapsing
    literal newlines is not enough on its own.

    `<` is escaped because this also renders into a Markdown sheet, where raw
    HTML is live and `<img src>` is the beacon `!url!` would have been.
    """
    # str() because a Jira field can be present-and-null, and this runs outside
    # the per-ticket try - a TypeError here would escape after the live writes.
    flattened = " ".join(str(text or "").split()).replace("\\", "")
    for char in ("[", "!", "<"):
        flattened = flattened.replace(char, "\\" + char)
    return flattened


def comment_body(cfg: dict, c) -> str:
    label = cfg["labels"][c.label]
    lines = [f"AI triage: {{{{{label}}}}}", "", wiki_safe(c.rationale)]
    if c.label == "needs_more_info" and c.missing_info:
        lines += ["", "Missing information:"] + [f"- {wiki_safe(m)}" for m in c.missing_info]
    if c.label == "automation_candidate" and c.verification_steps:
        lines += ["", "How to verify:"] + [f"- {wiki_safe(v)}" for v in c.verification_steps]
    lines += [
        "",
        "_Applied by the triage pilot bot from this ticket's visible content only. "
        f"Removing the label opts the ticket out of the pilot. (prompt {cfg['prompt']['version']})_",
    ]
    return "\n".join(lines)


def write_proposals(cfg: dict, out: pathlib.Path, stamp: datetime.datetime, proposals: list,
                    live: bool, source: str = "api") -> pathlib.Path:
    # Seconds included: two quick --keys runs in the same minute would
    # otherwise overwrite a grading sheet someone had already started on.
    base = out / f"proposals-{stamp:%Y%m%d-%H%M%S}"
    url = cfg["jira"]["base_url"] + "/browse/"
    with open(base.with_suffix(".csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(PROPOSAL_COLUMNS)
        for issue, c, chash in proposals:
            w.writerow(
                [issue["key"], url + issue["key"],
                 csv_safe(issue["fields"].get("summary", "")), c.label,
                 f"{c.confidence:.2f}", csv_safe(c.rationale),
                 csv_safe("; ".join(c.missing_info)),
                 csv_safe("; ".join(c.verification_steps)), chash, source, "", "", ""]
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
        # Escaped for the same reason as the comment body: `![](url)` in a
        # rationale is a tracking beacon fetched by everyone who opens this file.
        for issue, c, _ in proposals:
            fh.write(f"## [{issue['key']}]({url}{issue['key']}) "
                     f"{wiki_safe(issue['fields'].get('summary', ''))}\n\n")
            fh.write(f"**{c.label}** (confidence {c.confidence:.2f})\n\n"
                     f"{wiki_safe(c.rationale)}\n\n")
            if c.missing_info:
                fh.write("Missing info: " + "; ".join(map(wiki_safe, c.missing_info)) + "\n\n")
            if c.verification_steps:
                fh.write("Verification: " + "; ".join(map(wiki_safe, c.verification_steps)) + "\n\n")
    return base


def main(argv=None, out: pathlib.Path | None = None) -> int:
    ap = argparse.ArgumentParser(description="OpenMRS O3 AI triage pilot")
    ap.add_argument("--live", action="store_true", help="apply labels/comments (default: dry-run)")
    ap.add_argument("--limit", type=int, default=0, help="max tickets this run (0 = all)")
    ap.add_argument("--keys", help="comma-separated issue keys (skips the JQL sweep)")
    ap.add_argument("--no-classify", action="store_true", help="fetch and assemble contexts only")
    ap.add_argument("--classifications", metavar="JSON",
                    help="apply classifications from a file instead of calling Claude "
                         "(no Anthropic credential needed; see README)")
    ap.add_argument("--force", action="store_true",
                    help="reclassify already-triaged tickets (opt-outs are still respected)")
    ap.add_argument("--no-pr-check", action="store_true",
                    help="skip the GitHub open-PR backstop (offline runs; re-opens the "
                         "dev-panel gap documented in triage/github.py)")
    args = ap.parse_args(argv)

    cfg = load_config()

    # Resolved before any network call: a contradictory flag combination or an
    # unusable classifications file is a purely local fault, and failing on it
    # after a full JQL sweep wastes the sweep and buries the error under later
    # stdout. (Anthropic() resolves credentials lazily, so this does not surface
    # a missing key any earlier - only the local faults move.)
    if args.no_classify and args.classifications:
        sys.exit("--no-classify and --classifications are contradictory")
    classifier = None
    if args.classifications:
        classifier = FileClassifier(pathlib.Path(args.classifications),
                                    cfg["prompt"]["version"])
    elif not args.no_classify:
        classifier = Classifier(
            cfg["claude"]["model"], cfg["claude"]["max_tokens"],
            (ROOT / "prompt" / "system.md").read_text(),
        )

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

    # The dev-panel backstop. Off by config or by flag, and reported either way:
    # a sweep whose scope is wider than the pilot documented must say so in its
    # own log, not only in whatever the operator remembered to pass.
    gh_cfg = cfg.get("github") or {}
    github = None
    if gh_cfg.get("check_open_prs", False) and not args.no_pr_check:
        github = GitHubClient(gh_cfg.get("org", "openmrs"), os.environ.get("GITHUB_TOKEN"))
        auth = ("GITHUB_TOKEN" if github.authenticated else
                f"unauthenticated, {github.min_interval:.0f}s between searches")
        print(f"open-PR backstop: on (org {github.org}, {auth})")
    else:
        print("open-PR backstop: OFF; tickets with an open PR that the Jira dev "
              "panel missed will be classified")

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
            dev_clause = cfg["jira"]["dev_panel_clause"]
            if dev_clause in jql and "development" in str(e).lower():
                print("WARN: development[] JQL clause rejected here; sweeping without the "
                      "no-linked-PRs filter", file=sys.stderr)
                keys = jira.search_keys(jql.replace(dev_clause, ""))
            else:
                raise
    if args.limit:
        keys = keys[: args.limit]
    print(f"{len(keys)} ticket(s) in scope")

    source = "file" if isinstance(classifier, FileClassifier) else "api"
    if isinstance(classifier, FileClassifier):
        print(f"replaying {len(classifier.by_hash)} classification(s) by "
              f"{classifier.model}; no Claude call will be made")

    fields = ctx.ISSUE_FIELDS + ([ac_field] if ac_field else [])
    proposals: list = []
    manifest: dict = {}
    errors = consecutive = 0
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
            # Asymmetric on purpose. The pinned-model path must not skip a label
            # a replay produced, or one replay run would pin those tickets for
            # the rest of the pilot. The reverse must NOT hold: if a file run
            # also re-did api-labelled tickets, the two paths would flip the
            # label back and forth - and comment to every watcher - on every
            # sweep, since the documented workflow has both touching the cohort.
            stored_source = prop.get("source", "api")
            superseded = source == "api" and stored_source != "api"
            unchanged = (
                not args.live
                or (prop.get("contentHash") == chash
                    and prop.get("prompt") == cfg["prompt"]["version"]
                    and not superseded)
            )
            # Scope is re-checked only for live writes: a ticket can transition
            # out of "To Do" between the JQL sweep and this fetch, and labelling
            # it then invites a removal - which is a permanent opt-out and counts
            # against the removal-rate metric. Dry-run stays permissive so any
            # ticket can still be inspected with --keys.
            status = (issue["fields"].get("status") or {}).get("name")
            out_of_scope = args.live and status != cfg["jira"]["scope_status"]
            # plan_ticket stays the single authority on precedence; this only
            # avoids paying for a rate-limited GitHub search whose answer cannot
            # change the outcome. Unlike the status re-check above, the open-PR
            # check applies to dry runs too: it decides what reaches the grading
            # sheet, not just what gets written.
            provisional = plan_ticket(st, unchanged, args.force, classifier is not None,
                                      out_of_scope)
            open_prs: list[str] = []
            if github and provisional not in ("skip-opted-out", "skip-out-of-scope"):
                open_prs = github.open_pr_urls(key)
            action = plan_ticket(st, unchanged, args.force, classifier is not None,
                                 out_of_scope, bool(open_prs))
            # Excluded only for the three reasons that are permanent: an opt-out
            # the pilot promised to honour, a ticket that has left scope, and one
            # that is already in review.
            # "already-triaged" is NOT permanent - a live run re-classifies it
            # after a prompt bump or an edit - and the gather step is a dry run
            # where `unchanged` is unconditionally True, so gating on it dropped
            # every labelled ticket and left the re-triage backlog unclassified.
            if action not in ("skip-opted-out", "skip-out-of-scope", "skip-open-pr"):
                manifest[key] = {
                    "content_hash": chash,
                    "summary": issue["fields"].get("summary", ""),
                    "context": f"contexts/{key}.txt",
                }
            if action:
                row["action"] = action
                if action == "skip-opted-out":
                    row["by"] = st.opted_out_by
                elif action == "skip-out-of-scope":
                    row["status"] = status
                elif action == "skip-open-pr":
                    # Journalled so the exclusion is auditable: this is the one
                    # skip whose evidence lives outside Jira entirely.
                    row["open_prs"] = open_prs
                elif action == "skip-already-triaged":
                    row["labels"] = st.ai_labels_present
            else:
                c = classifier.classify(text)
                if c.refused:
                    row["action"] = "error-refusal"
                else:
                    row.update(action="proposed", label=c.label, confidence=c.confidence, model=c.model)
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
                            # `source` is set by the pipeline, so it is evidence.
                            # `classifier` is self-declared by whatever produced
                            # the classification and is a label, not proof - a
                            # file can claim any model name it likes.
                            "source": source,
                            "classifier": c.model,
                            "at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                        })
                        row["action"] = "labeled" if post_comment else "refreshed"
                    # Recorded only once the writes have landed: the live sheet
                    # is headed "labels and comments applied", so a ticket whose
                    # write raised must not appear in it.
                    proposals.append((issue, c, chash))
        except NotClassified as e:
            # Not a fault: a classifications file may cover a subset of the
            # sweep. Recorded as a skip so it never trips the error breaker.
            row["action"] = "skip-unclassified"
            row["detail"] = str(e)[:200]
        except Exception as e:
            row["action"] = "error"
            row["error"] = f"{type(e).__name__}: {e}"[:300]
        if st and st.human_adds:
            row["convention_violation_adds"] = st.human_adds
        row["at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        # An errored ticket writes no property, so its content hash never
        # changes and the next sweep classifies it again - a systemic fault
        # (a missing entity-property permission, a rotated API key, truncation)
        # would otherwise re-spend on every ticket every four hours, silently.
        # Stopping early caps that at CONSECUTIVE_ERROR_LIMIT paid calls.
        if row["action"].startswith("error"):
            errors += 1
            consecutive += 1
        else:
            consecutive = 0
        with open(journal, "a") as fh:
            fh.write(json.dumps(row) + "\n")
        suffix = f" -> {row['label']}" if row.get("label") else ""
        if row["action"] == "error":
            suffix = f" ({row['error']})"
        print(f"  {key}: {row['action']}{suffix}")
        if consecutive >= CONSECUTIVE_ERROR_LIMIT:
            print(f"\nABORT: {consecutive} tickets in a row failed; stopping the sweep "
                  "rather than re-spending on every remaining ticket", file=sys.stderr)
            break

    # Written only by the gather step. An apply run reaches a different (often
    # narrower) set of tickets, and overwriting the manifest with that set
    # destroyed the very description the next batch is built from.
    if args.no_classify and manifest:
        (out / "manifest.json").write_text(json.dumps({
            "at": stamp.isoformat(),
            "prompt_version": cfg["prompt"]["version"],
            # False if ANY ticket failed, not just the trailing run of them: an
            # errored ticket is simply absent from tickets[], so a consumer
            # trusting `complete` would never classify it.
            "complete": errors == 0 and (not args.limit or len(keys) < args.limit),
            # The schema of a classifications-file ENTRY, not of the model's
            # response: an entry additionally requires content_hash, and the
            # magnitude caps cannot be expressed in a response schema. A
            # consumer that validated against the response schema would produce
            # a file this pipeline rejects.
            "entry_schema": classification_entry_schema(),
            "tickets": manifest,
        }, indent=2) + "\n")
        print(f"\nManifest: {out / 'manifest.json'} ({len(manifest)} ticket(s) to classify)")

    if proposals:
        base = write_proposals(cfg, out, stamp, proposals, args.live, source)
        print(f"\nGrading sheet: {base}.csv (and .md); contexts in {out / 'contexts'}")
    if errors:
        # Non-zero so a scheduled run turns red. Previously a sweep in which
        # every ticket failed still exited 0, so the only evidence was a line
        # inside an artifact nobody opens.
        print(f"\n{errors} of {len(keys)} ticket(s) failed", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
