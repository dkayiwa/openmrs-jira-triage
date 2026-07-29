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
- **One `ai-triage-*` label + one comment per ticket, nothing else.** Writes
  live in one function, gated behind `--live`; the bot's permission scheme
  (below) backs this up.
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
export ANTHROPIC_API_KEY=...          # or have an `ant auth login` profile

.venv/bin/python -m triage.preflight  # connectivity, statuses, AC field, scope JQL
.venv/bin/python -m triage.run --limit 5   # dry-run: proposals + contexts + journal
```

Useful flags: `--keys O3-4522,O3-5823` (specific tickets), `--no-classify`
(context assembly only), `--force` (reclassify already-labeled tickets in
dry-run), `--live` (real writes; needs `JIRA_EMAIL`, `JIRA_API_TOKEN`,
`TRIAGE_BOT_ACCOUNT_ID`).

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
3. **Bot account** ("OpenMRS Triage Bot"): API token, and a permission scheme
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
7. **Weekly:** `python -m triage.metrics` prints the three pilot metrics and a
   draft ADOPT / EXTEND / STOP against the pre-registered thresholds. Pair with
   a native Jira filter subscription for the dashboard digest.

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
out/                  journal.jsonl, contexts/, proposals (gitignored)
```
