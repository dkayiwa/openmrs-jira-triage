"""Triage pilot runner.

Dry-run is the default: it writes contexts, a journal, the grading sheet
(.csv/.md) and the comment report (.html) under out/, and touches nothing in
Jira. --live applies at most one ai-triage-* label per ticket, plus one comment
the first time that label appears (a re-classification reaching the same label
is silent),
and is gated on three things: the credentials, the configured bot account id
actually matching them, and no scheduled sweep being able to race this one
(see schedule_conflict).

Scope is Jira's dev panel plus a GitHub backstop: a ticket whose PR the panel
has not indexed is still excluded if an open PR names its key.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import html
import json
import os
import pathlib
import sys
import tomllib
import unicodedata

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
    for line in env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def load_config() -> dict:
    _load_dotenv()
    with open(ROOT / "config.toml", "rb") as fh:
        return tomllib.load(fh)


def cohort_since_error(value, today: datetime.date | None = None) -> str | None:
    """Why this cohort window is unusable, or None.

    `cohort_created_since` is a bare date typed by hand and interpolated
    straight into JQL, and both ways of getting it wrong are silent. Jira does
    not reject an impossible date: measured, `2026-14-01` returns zero results
    rather than an error, and so does a year typed one ahead. Zero results then
    reaches preflight's three-state scope check, which compares the count with
    and without the development[] clause to tell "the clause is not being
    evaluated" from "the cohort is empty" - and both counts are zero, so it
    concludes the cohort is genuinely empty and PASSES.

    The sweep then does nothing, every four hours, and the gate says it is
    fine. The check is not wrong about what it measures; the cohort really is
    empty. It is the question that is wrong, because an empty cohort and an
    unanswerable query look identical from inside the query.

    A future window is refused outright: this value is launch minus ninety
    days by construction, so unlike pilot_launch there is no reading of it that
    is legitimately ahead of today.
    """
    if not value or not isinstance(value, str):
        return f"cohort_created_since is {value!r}; it must be a date like 2026-04-29"
    try:
        date = datetime.date.fromisoformat(value)
    except ValueError:
        return (f"cohort_created_since is {value!r}, which is not a real date. Jira "
                "accepts it and returns nothing, so the sweep would find an empty "
                "cohort and the gate would call it genuinely empty")
    today = today or datetime.datetime.now(datetime.timezone.utc).date()
    if date > today:
        return (f"cohort_created_since is {value}, which is after today ({today}). No "
                "ticket can have been created in that window, so the cohort is empty "
                "by construction and every sweep would do nothing")
    return None


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


def github_from_env(cfg: dict) -> GitHubClient | None:
    """The open-PR backstop client, or None when config disables it.

    Single-sourced for the same reason as ai_label_names: the sweep and preflight
    both build this, and a divergence would let preflight verify a backstop the
    sweep is not running (or the reverse), which is exactly the check nobody
    would think to make.
    """
    gh_cfg = cfg.get("github") or {}
    if not gh_cfg.get("check_open_prs", False):
        return None
    return GitHubClient(gh_cfg.get("org", "openmrs"), os.environ.get("GITHUB_TOKEN"))


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
            doc = json.loads(path.read_text(encoding="utf-8"))
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


WORKFLOW = ".github/workflows/triage.yml"
SCHEDULE_OVERRIDE = "--i-paused-the-schedule"


def schedule_conflict(root: pathlib.Path, in_ci: bool) -> str | None:
    """Why a local --live run is unsafe right now, or None if it is not.

    The workflow's `concurrency` group only serialises runs inside Actions, so it
    cannot see a run started from a laptop. Two sweeps that both read a ticket
    before either labels it will each post a comment to every watcher, and Jira
    Cloud has no way to un-send those - which is why this is a refusal rather
    than the README warning it used to be.

    Only local runs are gated; inside Actions the concurrency group already does
    the job. Fails CLOSED when the workflow cannot be read, because "probably no
    cron" is not worth duplicate comments on up to 32 public tickets. A missing
    file is not ambiguous, though: no workflow means no cron to race.
    """
    if in_ci:
        # Safe on the strength of the workflow's `concurrency` group, which this
        # function never reads. That is sound only because the group is pinned by
        # test_sweeps_cannot_overlap and the workflow runs the suite before the
        # sweep (test_the_suite_gates_the_sweep) - remove the group and CI fails
        # before it can write anything. Both tests are load-bearing here.
        return None
    path = root / WORKFLOW
    if not path.exists():
        return None
    try:
        # Imported here, not at module scope: only a live run needs it, and a
        # dry-run on a machine without PyYAML should still work.
        import yaml

        doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        # YAML reads a bare `on:` as the boolean True, so the triggers live under
        # either key depending on whether the file quotes it.
        triggers = doc.get(True) or doc.get("on") or {}
        active = "schedule" in triggers
    except Exception as e:
        return (f"cannot tell whether a scheduled sweep is enabled: reading "
                f"{WORKFLOW} raised {type(e).__name__}: {e}. Confirm none is "
                f"running and pass {SCHEDULE_OVERRIDE}")
    if not active:
        return None
    return (f"{WORKFLOW} has an active schedule, so a sweep may fire while this "
            "one runs. The workflow's concurrency group only serialises runs "
            "inside Actions - it cannot see this one, and two sweeps that both "
            "read a ticket before either labels it will each comment to every "
            "watcher, which Jira cannot un-send. Comment the schedule out (or "
            f"wait out the window), then pass {SCHEDULE_OVERRIDE}")


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
    #
    # Unicode format characters (category Cf) are dropped: bidi overrides
    # (U+202E) reorder everything after them, and zero-width characters hide
    # inside otherwise-innocent text. Both survive `.split()` because neither
    # is whitespace - every whitespace variant, including U+2028 and U+00A0,
    # is already collapsed by it. The footer telling maintainers that removing
    # the label opts the ticket out sits directly after the rationale in the
    # same comment, so an unterminated override lands on exactly the sentence
    # the opt-out guarantee depends on.
    #
    # Found by fuzzing rather than by a live exploit, and worth being exact
    # about: the one attempt to make the model carry U+202E from a ticket
    # summary into its rationale did NOT reproduce it. This is the backstop
    # whose stated premise is that model output is untrusted, so it should not
    # depend on that attempt having been representative. It costs the emoji
    # zero-width joiner, which triage prose has no use for.
    #
    # Backslashes are dropped BEFORE whitespace is collapsed: doing it after
    # left a token that was only backslashes behind as a doubled space, which
    # also made this function non-idempotent.
    cleaned = "".join(c for c in str(text or "")
                      if unicodedata.category(c) != "Cf").replace("\\", "")
    flattened = " ".join(cleaned.split())
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
        # "Permanently, and putting it back does not undo that" is the part
        # that has to be here rather than only in the announcement. This
        # comment is what a maintainer reads at the moment they decide, and
        # most will never have seen the announcement. Measured: remove the
        # label and restore it a minute later and the ticket is still opted
        # out (the removal is in the changelog forever, and --force does not
        # override it), while the restore is additionally recorded as a
        # convention violation and names them in the weekly digest. Saying
        # only "opts the ticket out" invites a tidy-up that cannot be undone,
        # on the pilot whose kill metric is people removing labels.
        "_Applied by the triage pilot bot from this ticket's visible content only. "
        "Removing the label opts this ticket out of the pilot permanently - putting "
        "the label back does not undo it - so please remove it to say the triage was "
        f"wrong, not to tidy up. (prompt {cfg['prompt']['version']})_",
    ]
    return "\n".join(lines)


def proposal_base(out: pathlib.Path, stamp: datetime.datetime) -> pathlib.Path:
    """Shared stem for a run's artifacts.

    Seconds included: two quick --keys runs in the same minute would otherwise
    overwrite a grading sheet someone had already started on. Hoisted out of
    write_proposals because the comment report is written on runs that produce
    no grading sheet at all, and the two must still share a stamp.
    """
    return out / f"proposals-{stamp:%Y%m%d-%H%M%S}"


def write_proposals(cfg: dict, out: pathlib.Path, stamp: datetime.datetime, proposals: list,
                    live: bool, source: str = "api") -> pathlib.Path:
    base = proposal_base(out, stamp)
    url = cfg["jira"]["base_url"] + "/browse/"
    with open(base.with_suffix(".csv"), "w", newline="", encoding="utf-8") as fh:
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
    with open(base.with_suffix(".md"), "w", encoding="utf-8") as fh:
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


REPORT_CSS = """
:root { color-scheme: light dark; --fg:#1a1a1a; --bg:#fff; --muted:#5c5c5c;
        --line:#d8d8d8; --pre-bg:#f6f6f4; --accent:#0f62fe; }
@media (prefers-color-scheme: dark) {
  :root { --fg:#e8e8e8; --bg:#161616; --muted:#a0a0a0; --line:#393939;
          --pre-bg:#202020; --accent:#78a9ff; }
}
* { box-sizing: border-box; }
body { margin:0 auto; padding:2rem 1.25rem 5rem; max-width:52rem; background:var(--bg);
       color:var(--fg); font:16px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI",
       Roboto, Helvetica, Arial, sans-serif; }
h1 { font-size:1.6rem; line-height:1.2; margin:0 0 .5rem; }
h2 { font-size:1.15rem; margin:2.5rem 0 .75rem; padding-bottom:.3rem;
     border-bottom:1px solid var(--line); }
h3 { font-size:1rem; margin:2rem 0 .4rem; }
a { color:var(--accent); }
p, li { margin:.5rem 0; }
.lede, .meta { color:var(--muted); font-size:.9rem; }
.note { border-left:3px solid var(--accent); padding:.6rem .9rem; margin:1.25rem 0;
        background:var(--pre-bg); border-radius:0 4px 4px 0; font-size:.92rem; }
table { border-collapse:collapse; width:100%; margin:1rem 0; font-size:.92rem; }
th, td { text-align:left; padding:.4rem .6rem; border-bottom:1px solid var(--line); }
th { font-weight:600; }
td.num { text-align:right; font-variant-numeric:tabular-nums; }
code { font-family:ui-monospace, SFMono-Regular, Menlo, monospace; font-size:.88em;
       background:var(--pre-bg); padding:.1em .35em; border-radius:3px; }
pre { background:var(--pre-bg); border:1px solid var(--line); border-radius:6px;
      padding:.9rem 1rem; overflow-x:auto; white-space:pre-wrap;
      word-wrap:break-word; font-family:ui-monospace, SFMono-Regular, Menlo, monospace;
      font-size:.85rem; line-height:1.5; }
.tag { display:inline-block; font-size:.78rem; padding:.1rem .45rem; border-radius:3px;
       background:var(--pre-bg); border:1px solid var(--line);
       font-family:ui-monospace, SFMono-Regular, Menlo, monospace; }
.wrap { overflow-x:auto; }
"""


def write_comment_report(cfg: dict, base: pathlib.Path, stamp: datetime.datetime,
                         proposals: list, live: bool, source: str,
                         excluded: list[dict], swept: int | None = None,
                         errors: int = 0) -> pathlib.Path:
    """Every label and comment this run writes (or would write), as one page.

    The reviewable artifact: Dennis and Veronica sign off on the wording before
    it reaches a public ticket. Rendered from the same comment_body() and
    plan_label_writes() the live path calls, so it is what Jira receives rather
    than a restatement of it - a report assembled independently could agree with
    the intent and still differ from the bytes.

    Everything interpolated is html.escape()d. Ticket text is untrusted, so
    model output derived from it is untrusted too: wiki_safe already defuses the
    Jira constructs, but this file is opened in a browser, where an unescaped
    <img src> in a rationale would be a tracking beacon fired at every reviewer.
    """
    ai_labels = ai_label_names(cfg)
    url = cfg["jira"]["base_url"] + "/browse/"
    rows = []
    for issue, c, _ in proposals:
        label = cfg["labels"][c.label]
        present = [l for l in issue["fields"].get("labels") or [] if l in ai_labels]
        add, remove, post_comment = plan_label_writes(present, label)
        rows.append({"issue": issue, "c": c, "label": label, "add": add,
                     "remove": remove, "post_comment": post_comment})

    def esc(text) -> str:
        return html.escape(str(text or ""))

    verb = "wrote" if live else "would write"
    parts = [
        "<!doctype html>", '<html lang="en"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f"<title>Triage pilot: what it {esc(verb)} - {stamp:%Y-%m-%d %H:%M} UTC</title>",
        f"<style>{REPORT_CSS}</style></head><body>",
        f"<h1>What the triage pilot {esc(verb)}</h1>",
        f'<p class="lede">{stamp:%Y-%m-%d %H:%M} UTC &middot; prompt '
        f'<code>{esc(cfg["prompt"]["version"])}</code> &middot; '
        f'{"LIVE - labels and comments applied" if live else "dry run - nothing written"}'
        f' &middot; {len(rows)} ticket(s) labelled'
        + (f", {len(excluded)} excluded as already in review" if excluded else "")
        + (f" &middot; {swept} in scope" if swept is not None else "")
        + "</p>",
    ]
    # The lede's "N in scope" is the honest denominator; a reader can see the gap
    # for themselves. Only errors get a warning box, because the arithmetic gap
    # is normally deliberate: in steady state most of the cohort is already
    # labelled and skipped, so warning on "labelled + excluded < swept" would
    # fire on every routine sweep and train operators to ignore the banner.
    # Errors are the signal - they are also the only way the sweep truncates,
    # since the breaker trips on consecutive failures.
    if errors:
        parts.append('<div class="note"><strong>This run did not complete '
                     f'cleanly.</strong> {errors} of {swept} ticket(s) errored, and '
                     "the sweep stops early if enough fail in a row - so tickets "
                     "may be missing from this report entirely. Check "
                     "<code>out/journal.jsonl</code> for the per-ticket outcome "
                     "before treating this as a complete record.</div>")
    if not live:
        parts.append(
            '<p class="meta">Every comment below is rendered by the same '
            "<code>comment_body()</code> the live path calls, and every label "
            "decision by the same <code>plan_label_writes()</code>, so this is the "
            "text Jira would receive verbatim.</p>")
    if source != "api":
        # Only `source` is evidence; the classifier string is self-declared.
        parts.append(
            '<div class="note"><strong>Not the pilot\'s measured path.</strong> These '
            f"classifications were replayed from a file (<code>source={esc(source)}</code>, "
            f"classifier <em>{esc(rows[0]['c'].model if rows else '?')}</em>), not produced "
            "by the pinned model through the API. The three pre-registered metrics assume "
            "one pinned model and prompt per label, so <code>triage.metrics</code> "
            "refuses to print a decision while such labels are in the cohort. Use this to "
            "review the rubric and the wording.</div>")

    counts = {key: 0 for key in LABEL_KEYS}
    for r in rows:
        counts[r["c"].label] += 1
    commented = sum(1 for r in rows if r["post_comment"])
    parts += ["<h2>Summary</h2>", '<div class="wrap"><table>',
              "<tr><th>Label</th><th>Tickets</th><th>Share</th></tr>"]
    for key in LABEL_KEYS:
        share = f"{counts[key] / len(rows) * 100:.0f}%" if rows else "-"
        parts.append(f'<tr><td><code>{esc(cfg["labels"][key])}</code></td>'
                     f'<td class="num">{counts[key]}</td><td class="num">{share}</td></tr>')
    parts += [f'<tr><th>total</th><th class="num">{len(rows)}</th><th></th></tr>',
              "</table></div>",
              f"<p>{commented} of {len(rows)} {'received' if live else 'would receive'} a "
              "comment as well as a label. A comment accompanies only a label that is new "
              "to its ticket, so a re-run that reaches the same label is silent.</p>"]

    if excluded:
        parts += ["<h2>Excluded: already in review</h2>",
                  "<p>No label, no comment. The Jira dev panel reported no pull request "
                  "for these, so <code>scope_jql</code> returned them; the open-PR "
                  "backstop caught them.</p>", "<ul>"]
        for row in excluded:
            # Linked only when it is a link. github.py emits a plain-text
            # stand-in ("openmrs PR #123 (no URL returned)") when the API
            # returns neither html_url nor url, and says why in as many words:
            # a synthesised reference "reaches the report as an href and
            # renders as a link that goes nowhere, which is worse than plain
            # text a reviewer can search for". Wrapping every entry in an
            # anchor defeated that - the stand-in became
            # <a href="openmrs PR #123 (no URL returned)">, a relative href
            # that 404s - so the producer's care was undone by the consumer.
            # This is the evidence list for tickets held out of scope, which is
            # exactly where a reviewer follows the link to check the claim.
            links = " ".join(
                f'<a href="{esc(u)}">{esc(u)}</a>' if u.startswith(("http://", "https://"))
                else f"<code>{esc(u)}</code>"
                for u in row["open_prs"])
            parts.append(f'<li><a href="{url}{esc(row["key"])}">{esc(row["key"])}</a> '
                         f'{esc(row["summary"])} &mdash; {links}</li>')
        parts.append("</ul>")

    if rows:
        parts.append("<h2>The comments</h2>")
    for r in rows:
        issue, c = r["issue"], r["c"]
        key = issue["key"]
        removed = (" &middot; removed: " +
                   ", ".join(f'<span class="tag">{esc(l)}</span>' for l in r["remove"])
                   if r["remove"] else "")
        no_comment = "" if r["post_comment"] else (
            " &middot; label already present, so no comment")
        parts += [
            f'<h3><a href="{url}{esc(key)}">{esc(key)}</a> '
            f'{esc(issue["fields"].get("summary"))}</h3>',
            f'<p class="meta"><span class="tag">{esc(r["label"])}</span> &middot; '
            f"confidence {c.confidence:.2f}{removed}{no_comment}</p>",
            f"<pre>{esc(comment_body(cfg, c))}</pre>",
        ]
    parts.append("</body></html>")
    report = base.with_suffix(".html")
    report.write_text("\n".join(parts), encoding="utf-8")
    return report


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
    ap.add_argument("--i-paused-the-schedule", action="store_true",
                    help="for a local --live run while the workflow schedule is "
                         "enabled: asserts no scheduled sweep can fire concurrently")
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
            (ROOT / "prompt" / "system.md").read_text(encoding="utf-8"),
        )

    jira = jira_from_env(cfg)
    bot_id = os.environ.get("TRIAGE_BOT_ACCOUNT_ID")
    if args.live and not (jira.authenticated and bot_id):
        sys.exit("--live needs JIRA_EMAIL, JIRA_API_TOKEN and TRIAGE_BOT_ACCOUNT_ID")
    # Checked before bot_identity_error's request: this is a purely local fault,
    # and the operator should learn about it without waiting on the network.
    if args.live and not args.i_paused_the_schedule:
        clash = schedule_conflict(ROOT, os.environ.get("GITHUB_ACTIONS") == "true")
        if clash:
            sys.exit(clash)
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
    github = None if args.no_pr_check else github_from_env(cfg)
    if github:
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
        window_error = cohort_since_error(cfg["jira"]["cohort_created_since"])
        if window_error:
            sys.exit(f"config.toml [jira]: {window_error}")
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
    # Carried into the comment report: a reviewer seeing 32 tickets where the
    # sweep said 34 needs to know which two were held back and on what evidence,
    # not just notice that they are missing.
    excluded: list[dict] = []
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
            (out / "contexts" / f"{key}.txt").write_text(text, encoding="utf-8")
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
            # The property is OUR bookkeeping, but it lives in Jira, so anyone
            # with API access to the issue can put anything under this key -
            # and the key is in a public repo. A non-dict raised AttributeError
            # on the .get below, which errored the ticket; five such tickets
            # tripped the consecutive-error breaker and aborted the sweep. The
            # scope JQL orders by created ASC, so the same five would be met
            # first on every run: the pilot stops, permanently, and the only
            # trace is a journal nobody reads. metrics.py already guards this
            # exact read - "hand-set, or written by something else" - because
            # the same thought occurred there and not here.
            #
            # Treated as absent rather than fatal: unknown bookkeeping means we
            # cannot claim the ticket is unchanged, so it is re-classified and
            # the property is rewritten. That costs one classification and
            # repairs itself, where failing costs the whole sweep forever.
            if not isinstance(prop, dict):
                print(f"WARN: {key}: entity property {PROPERTY_KEY} is "
                      f"{type(prop).__name__}, not an object - something other than "
                      "this pipeline wrote it; treating the ticket as untriaged",
                      file=sys.stderr)
                prop = {}
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
            # Consulted only where the answer can change something. It cannot for
            # a ticket already skipped for a permanent reason; and outside the
            # gather step it cannot for an already-triaged one either, because
            # the manifest the answer would alter is written under --no-classify
            # only. Gather still asks - it deliberately offers already-triaged
            # tickets for re-classification, so one in review must drop out
            # there. Skipping the rest matters: in steady state most of the
            # cohort is already triaged, and each search costs 2.5s of throttle
            # and one more chance of the secondary rate limit that fails a
            # ticket outright.
            deaf_to_answer = provisional in ("skip-opted-out", "skip-out-of-scope") or (
                provisional == "skip-already-triaged" and not args.no_classify)
            open_prs: list[str] = []
            if github and not deaf_to_answer:
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
                    excluded.append({"key": key, "open_prs": open_prs,
                                     "summary": issue["fields"].get("summary") or ""})
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
                        # Recorded here, before the property write. Everything a
                        # watcher can see has already landed; set_property is
                        # internal bookkeeping. Dropping the ticket when only
                        # that fails would hide a public comment from the page
                        # headed "what the triage pilot wrote" - and the property
                        # write is exactly the one preflight probes because it is
                        # the permission most likely to be missing.
                        proposals.append((issue, c, chash))
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
                    else:
                        # Dry run: nothing to land, so record it directly.
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
        with open(journal, "a", encoding="utf-8") as fh:
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

    base = proposal_base(out, stamp)
    # An empty grading sheet is noise, so that stays gated on proposals. The
    # comment report is not: a sweep the breaker cut short, or one whose only
    # outcome was open-PR exclusions, is exactly when its truncation banner and
    # excluded list need to exist.
    if proposals:
        write_proposals(cfg, out, stamp, proposals, args.live, source)
        print(f"\nGrading sheet: {base}.csv (and .md); contexts in {out / 'contexts'}")
    if proposals or excluded or errors:
        report = write_comment_report(cfg, base, stamp, proposals, args.live, source,
                                      excluded, swept=len(keys), errors=errors)
        wrote = "what was written" if args.live else "what would be written"
        print(f"\nComment report ({wrote}): {report}")
    if errors:
        # Non-zero so a scheduled run turns red. Previously a sweep in which
        # every ticket failed still exited 0, so the only evidence was a line
        # inside an artifact nobody opens.
        print(f"\n{errors} of {len(keys)} ticket(s) failed", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
