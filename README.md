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

## Quickstart (dry-run, no Jira credentials needed)

```sh
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
echo 'ANTHROPIC_API_KEY=sk-ant-...' > .env   # .env is gitignored and auto-loaded

.venv/bin/python -m unittest discover -s tests -q  # offline logic tests
.venv/bin/python -m triage.preflight  # connectivity, statuses, AC field, scope JQL
.venv/bin/python -m triage.run --limit 5   # dry-run: proposals + contexts + journal
```

Useful flags: `--keys O3-4522,O3-5823` (specific tickets), `--no-classify`
(context assembly only), `--force` (reclassify already-labeled tickets; in
`--live` this re-spends on classification and comments again on any label
flip, but opt-outs are still respected), `--live` (real writes; needs
`JIRA_EMAIL`, `JIRA_API_TOKEN`, `TRIAGE_BOT_ACCOUNT_ID`).

## Running without an Anthropic API key

The Claude call is the only step that needs an Anthropic credential — Jira reads
work anonymously on a public project. So the pipeline splits into three, and the
middle step can be done by whatever you like, including an agent in a Claude
Code session using your own account:

```sh
# 1. gather - no Anthropic credential, writes contexts + a manifest
.venv/bin/python -m triage.run --no-classify --limit 5

# 2. classify - read out/manifest.json (it embeds the JSON schema and the
#    content hash of each context) and write out/classifications.json

# 3. apply - replays those classifications; makes no Claude call
.venv/bin/python -m triage.run --classifications out/classifications.json --live
```

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

Lookup is by content hash, not ticket key, so staleness is caught for free: if
the ticket is edited between being classified and being applied, the freshly
assembled context hashes differently, no classification matches, and the stale
label is never written. A `prompt_version` that disagrees with `config.toml`,
a missing `content_hash`, or anything failing the schema aborts before any
write. The `classifier` string is recorded in each ticket's entity property so
an audit can tell replayed labels from pinned-model ones.

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

A dry run produces `out/proposals-<stamp>.md` (readable) and `.csv` (gradable).
They mark each row `ok` or `wrong` (plus `correct_label` when wrong), then:

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
triage/state.py       changelog-derived opt-out / already-triaged / violations
triage/classifier.py  one Claude call, JSON-schema output, refusal fallbacks
triage/preflight.py   pre-launch checks incl. label charset test
triage/metrics.py     live-phase metrics + decision rule
evals/run_evals.py    graded regression set for prompt changes
tests/                offline unit tests (no network, no API key)
out/                  journal.jsonl, contexts/, proposals (gitignored)
```

