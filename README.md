# openmrs-jira-triage

Working dry-run of the **O3 AI triage pilot** ("Triage pilot - plan and decisions",
Dennis Kigen & Veronica Muthee).

It is a deterministic pipeline with exactly one Claude call per ticket, not an
agent with Jira access. The model only returns a classification object; this
code does every Jira read and write, so the pilot's guarantees hold by
construction:

- **Visible information only.** Contexts are assembled in code from summary,
  description, Acceptance Criteria, parent, linked tickets, and comments
  filtered to human authors (`accountType != "app"` plus a blocklist). The
  exact text sent to the model is saved to `out/contexts/<KEY>.txt` for audit.
- **One `ai-triage-*` label at a time, one comment per label decision, nothing
  else.** Writes are gated behind `--live` and the bot's permission scheme
  (below). A re-run that reaches the same label is silent; only a label that is
  new to the ticket is commented on, so an edit that genuinely changes the
  classification posts a second comment explaining the new label rather than
  leaving it unexplained.
- **Label removal = permanent opt-out.** Detected from the issue changelog
  (any non-bot removal of an ai-triage label), so there is no external state
  to lose. Non-bot label *adds* are flagged as convention violations.
- **Pre-registered decisions.** Cohort window, thresholds, and prompt version
  live in `config.toml`, committed before launch.

> **Label names differ from the doc.** Jira Cloud rejects `/` in labels, so
> `ai-triage/automation-candidate` is configured as
> `ai-triage-automation-candidate` etc. Confirm with the preflight charset test.

## Scope: tickets already in review

`scope_jql` excludes tickets whose Jira dev panel lists a pull request, but that
panel only sees what the GitHub-for-Jira app indexed. On the launch cohort two
of the 34 tickets it returned were already in review:

| Ticket | Open PR the dev panel missed |
|---|---|
| O3-5796 | `openmrs-esm-smart-notifications-app#1` "(feat) O3-5796: Add Bell Notification Icon" |
| O3-5816 | `openmrs-esm-core#1818` "(fix) O3-5816: Stop LocationPicker crashing…" |

The JQL clause is not broken and does evaluate anonymously (2377 O3 tickets
match `.all > 0`) — the gap is the index, so no JQL edit closes it. Labelling a
ticket that is already in review invites the removal this pilot treats as a
**permanent opt-out**, and that removal counts against the kill metric. So the
sweep asks GitHub for the same evidence the dev panel is supposed to carry: an
**open PR naming the ticket key**, journalled as `skip-open-pr` with the PR URL.

Deliberate limits, both erring towards leaving a ticket in scope:

- **A key only in a PR *comment* does not count.** Search matches comments too,
  so "unrelated to O3-5816" in a review would otherwise exclude a ticket nobody
  is working on. Only the PR title and body are treated as a claim.
- **A PR that never names the ticket is invisible here.** O3-5843's description
  links `openmrs-esm-audit-log-app#1`, but that PR cites no key, so neither Jira
  nor GitHub can tie the two. The fix is the convention the other PRs already
  follow — put the key in the PR title. Prose-scanning the ticket instead would
  misfire: the same URL appears in O3-5842 as the place a styleguide bug was
  *observed*, and in O3-5828 as a dependency.
- **A closed PR does not count.** O3-5716's `bedmanagement#114` was closed
  unmerged on 2026-07-08, so that ticket genuinely is the pilot's to sort.

`GITHUB_TOKEN` is optional and read-only; it raises the search limit from 10 to
30/min (`GITHUB_TOKEN=$(gh auth token)` locally). The scheduled workflow uses
the automatic `github.token`, so no new repo secret is needed. A failed search
**fails that ticket** rather than classifying it — fail-open would silently
re-open this gap on exactly the tickets most likely to be in review. Use
`--no-pr-check` (or `[github].check_open_prs = false`) to sweep on Jira's word
alone; either way the run says which mode it is in.

## Quickstart (dry-run, no Jira credentials needed)

```sh
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
echo 'ANTHROPIC_API_KEY=sk-ant-...' > .env   # .env is gitignored and auto-loaded

.venv/bin/python -m unittest discover -s tests -q  # offline logic tests
.venv/bin/python -m triage.preflight  # connectivity, statuses, AC field, scope JQL
.venv/bin/python -m triage.run --limit 5   # dry-run: proposals + contexts + journal
open out/proposals-*.html                  # what it would write to each ticket
```

Useful flags: `--keys O3-4522,O3-5823` (specific tickets), `--no-classify`
(context assembly only), `--force` (reclassify already-labeled tickets; in
`--live` this re-spends on classification and comments again on any label
flip, but opt-outs are still respected), `--live` (real writes; needs
`JIRA_EMAIL`, `JIRA_API_TOKEN`, `TRIAGE_BOT_ACCOUNT_ID`), `--no-pr-check`
(sweep without the open-PR backstop below).

## Running without an Anthropic API key

The Claude call is the only step that needs an Anthropic credential — Jira reads
work anonymously on a public project. So the pipeline splits into three, and the
middle step can be done by whatever you like, including an agent in a Claude
Code session using your own account:

```sh
# 1. gather - no Anthropic credential, writes contexts + a manifest
.venv/bin/python -m triage.run --no-classify --limit 5

# 2. classify - read out/manifest.json (it lists every ticket eligible for
#    classification with its content hash, and embeds entry_schema, the schema
#    of one classifications-file entry) and write out/classifications.json

# 3. apply - replays those classifications; makes no Claude call
.venv/bin/python -m triage.run --classifications out/classifications.json --live
```

The apply sweep may be wider than the file: a ticket with no entry is journalled
`skip-unclassified` and left alone, so a file covering part of the cohort is a
normal way to work through it in batches. Only the gather step writes the
manifest, so an apply run cannot clobber the description the next batch is built
from.

`out/classifications.json` looks like this; `content_hash` comes from the
manifest and is required:

```json
{
  "prompt_version": "v1",
  "classifier": "who or what produced these",
  "classifications": {
    "O3-4522": {
      "content_hash": "15e75ed5e4489c82",
      "label": "needs_more_info",
      "rationale": "at most two sentences, posted to the ticket",
      "missing_info": ["which report was being run"],
      "verification_steps": [],
      "confidence": 0.93
    }
  }
}
```

Lookup is by content hash, so staleness is caught for free: if the ticket is
edited between being classified and being applied, the freshly assembled context
hashes differently and the stale label is never written. Each entry is also
checked against the `TICKET:` line of the context it matched, so two entries
whose `content_hash` values are mispaired are refused rather than applied to each
other's tickets. A `prompt_version` disagreeing with `config.toml`, a duplicate
or non-string `content_hash`, an empty rationale, a confidence outside 0–1, or
over-length text all abort before any Jira call.

Each ticket's entity property records **`source`** (`api` or `file`), set by the
pipeline, alongside a self-declared `classifier` string. Only `source` is
evidence — a file can claim any model name it likes.

`source` participates in the idempotency key **asymmetrically**: a live API sweep
treats a replayed label as stale and re-classifies it, so one replay run cannot
pin a ticket for the rest of the pilot — but a replay run leaves an API label
alone. Symmetry here would make the two paths swap the label and comment to every
watcher on every sweep, since this workflow has both touching the same cohort.

> **This is not the pilot's measured path.** The eval gate and the three
> pre-registered metrics assume one pinned model and prompt version per label.
> Replayed classifications are useful for grading, for spot-checking the rubric,
> and for running without API access — but mixing them into the live cohort
> makes the removal-rate metric a measurement of two different systems. The
> scheduled sweep in `.github/workflows/triage.yml` still uses
> `ANTHROPIC_API_KEY`, and has to: a cron run has no session to borrow.

For local runs that *do* use the API, a key is not the only option — the client
is constructed zero-arg, so an `ant auth login` profile is picked up
automatically with no `.env` at all.

## What to hand to Dennis & Veronica

Every run writes three files sharing one stamp:

| File | For |
|---|---|
| `proposals-<stamp>.html` | **Review before launch.** Every label and the exact comment body each ticket would receive, plus the tickets held back as already in review. Rendered by the same `comment_body()` and `plan_label_writes()` the live path calls, so it is the text Jira receives rather than a restatement of it. Written on live runs too, where it is an audit trail of what *was* posted. |
| `proposals-<stamp>.md` | Readable list of proposals |
| `proposals-<stamp>.csv` | Gradable sheet |

Send the HTML report round first — it is the artifact to sign off on, because
comments notify every watcher and Jira Cloud has no off switch. Then grade the
CSV: mark each row `ok` or `wrong` (plus `correct_label` when wrong), and:

```sh
.venv/bin/python evals/run_evals.py --import-proposals out/proposals-<stamp>.csv
.venv/bin/python evals/run_evals.py        # re-run after every prompt change
```

The graded set becomes the regression suite: iterate `prompt/system.md` (bump
`[prompt].version`) until agreement is >= 90%, the offline proxy for the
pilot's "fewer than 1 in 10 labels removed" kill metric. During the live
phase, every human label-removal gets added here as a new case.

## Going live checklist

1. **Status backfill first.** Scope depends on `status = "To Do"` being
   trustworthy; snapshot the cohort only after the clean-slate backfill.
2. **Pin the cohort.** Set `cohort_created_since` to launch minus 90 days and
   commit. Tickets created during the pilot enter scope automatically.
3. **Bot account** ("OpenMRS Triage Bot") — do not replace this account mid-pilot.
   Opt-out attribution keys on a single `TRIAGE_BOT_ACCOUNT_ID`, so swapping in a
   new account makes every label change the old one made read as a human removal:
   the whole cohort registers as opted out and the weekly decision flips to STOP.
   Rotate the API *token* freely; keep the account. It needs a permission scheme
   granting only Browse Projects, Edit Issues (labels need it), and Add
   Comments on O3 - no Transition, Delete, Assign, or Link. Note Edit Issues
   is not field-granular in Jira, so "never edits ticket text" is enforced by
   this code writing only the labels field; audit via changelogs weekly.
   `notifyUsers=false` on label edits additionally needs project admin, else
   edits notify watchers (comments always notify - Cloud has no off switch).
4. **Preflight with the write test:** `python -m triage.preflight --scratch O3-XXXX`.
5. **Dry-run the whole cohort, grade, iterate** until the eval gate passes.
6. **Enable writes:** add the four repo secrets, uncomment the `schedule` block
   in `.github/workflows/triage.yml`, or run `python -m triage.run --live`.
   (`GITHUB_TOKEN` is not a fifth secret — the workflow passes the automatic
   `github.token`, which `permissions: contents: read` already covers.)
7. **Weekly:** `python -m triage.metrics` prints the three pilot metrics, any
   non-bot `ai-triage-*` label adds (convention violations), and a draft
   ADOPT / EXTEND / STOP against the pre-registered thresholds, computed purely
   from Jira changelogs (set `pilot_launch` in config first). Pair with a native
   Jira filter subscription for the dashboard digest.
   Never run `--live` locally while the scheduled sweep may be running: the
   workflow's `concurrency` group only serialises runs inside Actions, and two
   sweeps that both read a ticket before either labels it will each post a
   comment to every watcher.
8. **Process note for maintainers:** removing an `ai-triage-*` label opts the
   ticket out permanently - including after promoting a ticket to `intro`.
   Leave the ai-triage label in place; the intro metric counts tickets holding
   both labels.

## Layout

```
config.toml           pre-registered pilot config (cohort, labels, thresholds)
prompt/system.md      the rubric prompt (versioned)
triage/run.py         sweep -> skip checks -> assemble -> classify -> write/propose
triage/context.py     visible-information assembly + content hash
triage/github.py      open-PR backstop for the Jira dev panel (scope)
triage/state.py       changelog-derived opt-out / already-triaged / violations
triage/classifier.py  one Claude call, JSON-schema output, refusal fallbacks
triage/preflight.py   pre-launch checks incl. label charset test
triage/metrics.py     live-phase metrics + decision rule
evals/run_evals.py    graded regression set for prompt changes
tests/                offline unit tests (no network, no API key)
out/                  journal.jsonl, contexts/, proposals .html/.md/.csv (gitignored)
```

