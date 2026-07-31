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
alone; either way the run says which mode it is in. **Pass the flag to
`triage.preflight` as well as `triage.run`** — the gate probes the same search
and runs as a required workflow step, so skipping it in one place and not the
other leaves the sweep blocked by a red gate. The config toggle needs saying
once and covers both.

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

This is the path to use when the Anthropic credit runs out: step 3 constructs no
Anthropic client at all, so a live run needs only the bot's Jira credentials. Two
things to know before relying on it. A local `--live` run is refused while the
workflow `schedule` block is active (see the go-live checklist) — comment it out
and pass `--i-paused-the-schedule`. And because each ticket records
`source=file`, `triage.metrics` will print NO DECISION and list those tickets
until a pinned-model sweep re-classifies them: the replay keeps the pilot moving,
but it does not produce measurable weeks.

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

**Comparing models** is a question the graded set can answer, once it exists:

```sh
.venv/bin/python evals/run_evals.py --model claude-sonnet-5 --model claude-haiku-4-5
```

`--model` is repeatable and scores the same frozen contexts against each one, so
"would a cheaper model do as well" stops being an argument. It is deliberately
**additive**: there is no way to make the *gate* measure anything but the model
`config.toml` pins, the comparison labels itself as not the gate, and its exit
code reports whether the comparison ran rather than whether anything was
cleared. The pinned pair is what the pilot committed to before seeing results;
a model shopped for afterwards must not inherit that authority. Note the
volumes involved are small — a full sweep is ~49k input tokens, measured over
the 31-ticket cohort under prompt v3 — so the saving from a cheaper model is a
few dollars a month against a kill metric that stops the pilot at four removals
out of ~31. Steady-state sweeps cost far less again, because an unchanged
ticket is skipped before it is ever sent.

   Worth knowing before optimising the wrong thing: **73% of that input is the
   system prompt, not the tickets.** It is 4,665 characters re-sent with all 31
   calls (~36k tokens) against ~13k for every context combined. So the lever on
   cost is prompt caching, not a cheaper model — and caching would leave the
   pinned pair, and therefore the pre-registered metrics, untouched. Not
   implemented here: it changes how the pipeline calls the API, which is the
   owners' call during a pre-registered pilot, not a hardening change.

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

   **Know what passing it proves.** The gate is a point estimate: each graded
   case is classified once, and 90% is compared against the fraction that came
   back. `evals/run_evals.py` now prints the interval around that fraction, and
   at cohort scale it is wide — an observed 90.3% over 31 cases carries a 95%
   interval of 75%–97%, so it does not separate a model that agrees nine times
   in ten from one that agrees three times in four. Only more *graded cases*
   narrow it; re-running the same ones does not. Two consequences: treat a
   near-90% pass as "not yet ruled out" rather than cleared, and prefer grading
   breadth over prompt micro-iteration once you are close, because past a point
   the gate can no longer see the difference you are making.

   **The classifier is not deterministic.** Measured by
   `evals/injection_eval.py`: the same ticket text, five runs, four
   `automation_candidate` and one `needs_judgment`. `temperature` is deprecated
   on this model, so it cannot be pinned. This bears on the kill metric in step
   7 — a re-classification after a content edit can move a label for no reason
   but the draw, which posts a fresh comment and counts as a removal if a
   maintainer prefers the first — and it means part of any agreement shortfall
   is the model disagreeing with itself rather than with the graders. Whether
   that is acceptable is a pilot-design call for the owners; it is recorded
   here so it is made deliberately.

   **Grade the contested tickets first.** Unanimity teaches you least — a case
   every model labels the same way was easy. Classify the frozen contexts with
   two or three candidate models and grade where they *split*: that set is both
   the model question and the largest pocket of prompt ambiguity, for a fraction
   of the effort of grading the whole cohort. Measured on the launch cohort,
   11 of 32 tickets split, and 6 of those 11 sat on the `needs_judgment`
   boundary — the label the rubric finds hardest.

   **Settle the model before launch, and don't shop for one after.** The three
   metrics assume one pinned model and prompt version per label, so the choice is
   pre-registered like the thresholds. Once grades exist,
   `evals/run_evals.py --model <id> --model <id>` scores candidates against them
   through the real classifier; that comparison is labelled as not the gate and
   cannot change what the pinned model authorises.

   **These numbers were measured under prompt v2, and the pinned prompt is now
   v3.** Read them as a lead, not as a settled result — the paragraph below
   originally said the next person need not re-derive it, and that was wrong
   the moment the rubric changed. `evals/run_evals.py` states the rule this
   broke, in the code: a result recording a model without its prompt "cannot be
   reproduced later". The v3 rewrite moved 7 of 31 tickets and specifically
   rewrote the `needs_judgment` definition — which is the exact label the Haiku
   finding turns on — so the direction below is plausible under v3 but the
   counts are not. Re-run `--model` against the graded set once cases exist,
   and replace this paragraph with the v3 figures and this sentence with the
   version they came from.

   What that v2 comparison found: **Haiku is unsuitable.** It agreed with Opus
   on 24/32, and all 8
   of its disagreements moved *toward* `automation_candidate` — six from
   `needs_judgment`, collapsing that label from 7 to 1. That is the harmful
   direction: it tells a contributor a ticket blocked on a clinical or product
   call is ready to build. Its confidence was also anti-informative, highest
   exactly where the models disagreed (0.83 on split cases vs Opus's 0.62), so it
   would flag nothing for review. **Sonnet was not resolved:** it differed on 5
   of 32, which is within about twice the harness effect measured alongside it,
   so that gap may be the test wrapper rather than the model. Note the stakes are
   asymmetric — the saving from a cheaper model is a few dollars a month against
   a kill metric that stops the pilot at four removals out of ~31. (Both figures
   restated above under "Running without an Anthropic API key"; they were two
   copies of one sentence and had already drifted apart by the time anyone
   checked, which is the argument for reading that section as the source.)
6. **Enable writes:** add the four repo secrets, uncomment the `schedule` block
   in `.github/workflows/triage.yml`, or run `python -m triage.run --live`.
   (`GITHUB_TOKEN` is not a fifth secret — the workflow passes the automatic
   `github.token`, which `permissions: contents: read` already covers.)
7. **Weekly:** `python -m triage.metrics` prints the three pilot metrics, any
   non-bot `ai-triage-*` label adds (convention violations), and a draft
   ADOPT / EXTEND / STOP against the pre-registered thresholds, computed purely
   from Jira changelogs (set `pilot_launch` in config first). Pair with a native
   Jira filter subscription for the dashboard digest.
   A local `--live` run while the scheduled sweep may fire is **refused**, not
   merely discouraged: the workflow's `concurrency` group only serialises runs
   inside Actions, and two sweeps that both read a ticket before either labels it
   will each post a comment to every watcher — which Jira Cloud cannot un-send.
   Comment the `schedule` block out, then pass `--i-paused-the-schedule`. The
   check is skipped inside Actions (where the concurrency group already does the
   job) and fails closed if the workflow file cannot be read.

   **Rehearse this before launch.** Most of `metrics.py` sits behind
   `if not st.bot_first_labeled_at: continue`, so until the bot has labelled
   something, a real run walks the whole cohort and exits without touching the
   property read, the 24h SLA, the removal count or the decision rule. Those have
   unit tests, but fixtures only prove what their author already believed.
   `metrics.py` has no opinion about *which* labels it measures, so point
   `[labels]` at a label already in use and `TRIAGE_BOT_ACCOUNT_ID` at whoever
   has been adding it, patch both in memory rather than on disk, and the same
   code walks the same shapes on real bytes. Read-only. The numbers are
   meaningless as pilot metrics; the code producing them is what runs live.

   **Set `pilot_launch` earlier than the labels you are rehearsing on** — an
   existing label was applied months or years ago, and `sla_met` refuses a
   label that predates the launch, because in production that means the launch
   date is mistyped and every ticket would otherwise report as sorted on time.
   In a rehearsal it means nothing of the sort, but the guard cannot tell the
   difference. Skip this and every ticket fails with "the bot cannot have
   labelled a ticket before the pilot began" and the run exits having computed
   nothing. Rehearsed on `curated` with `pilot_launch = "2024-01-01"`: 9
   tickets measured, SLA 11.1%, removal rate 0.000, and the decision correctly
   withheld because none carries an entity property.

   Two shapes are worth arranging deliberately, because they are what the
   guarantees rest on. A ticket the stand-in bot labelled and *someone else*
   unlabelled must count as an opt-out; a ticket where the stand-in both added
   and removed the label must **not** - that is the protection against the bot's
   own label flips reading as a cohort-wide opt-out. Done once against O3-3121
   (added by one person, removed by another) and O3-4115 (added and removed by
   the same person 31 seconds apart): 1 of 4, not 2 of 4. The run also settled an
   open question - anonymous entity-property reads 404 cleanly rather than 401,
   so a missing property reads as "never triaged" as intended.

   Out of reach this way, and left to their unit tests: the replayed-`source`
   warning (needs a property only a write can create) and the failed-read
   NO DECISION path. Both parse data this pipeline writes itself rather than
   anything Jira's shape can surprise us with.
8. **Tell the maintainers first**, using `docs/maintainer-announcement.md`
   (fill in the bracketed placeholders). Three asks, and none of them are
   optional: removing an `ai-triage-*` label opts the ticket out permanently, so
   it must not be done as housekeeping - including after promoting a ticket to
   `intro`, since the intro metric counts tickets holding both labels. Keys go in
   PR titles, or the open-PR backstop cannot see the PR. And `intro`/`not-intro`
   have to be applied to triaged tickets, or the third pre-registered threshold
   stays at zero and ADOPT is unreachable however good the triage is.

   Post it **after** the charset test in step 4: the announcement names the
   labels, and a rename after posting means the community is watching for a label
   that never arrives. The label names in it are pinned to `config.toml` by the
   test suite, so a rename breaks the build rather than the announcement.

## Verifying a change

```bash
.venv/bin/python -m unittest discover -s tests -q     # 378 tests, ~3s, no network
.venv/bin/python -m triage.preflight                  # the go-live gate, read-only
```

Both are offline except preflight, and CI runs both before any sweep. One more
is offline but slow enough to keep out of the suite, and two cost real API
calls:

```bash
.venv/bin/python -m evals.fuzz_sanitisers             # offline; after touching wiki_safe
.venv/bin/python -m evals.run_evals                   # the graded gate (needs cases)
.venv/bin/python -m evals.injection_eval              # run before a prompt change
```

**What is already verified, so you neither redo it nor assume it is missing.**
Every module is mutation-tested and branch coverage is 96%. The sanitisers are
fuzzed against their stated invariants by the script above, the HTML report is
checked by parsing it rather than by reading it for escapes, the opt-out
guarantee is exercised across sweeps rather than against a fixture, and
`injection_eval.py` tests the threat model `triage/classifier.py` asserts in
its docstring.

What the remaining 3% is, exactly: `sys.exit(main())`; one-line delegations to
`_check(_get(...))`; a session auth assignment; `continue`/`pass` inside
exception handlers; one `WARN` print. The rest is `preflight.py`'s `--scratch`
write probes, which cannot run without bot Jira credentials - the same blocker
as the go-live checklist's step 4, and the only genuinely untested behaviour
left in the pipeline.

This sentence was wrong when first written. It called the remainder inert
while `sys.exit(mismatch)` sat inside it - the line that aborts a live run
when `TRIAGE_BOT_ACCOUNT_ID` does not match the credentials, which is the
guard against the entire cohort reading as opted out, and it had never
executed. Check a claim like this against `coverage report -m`, not memory.

**Reaching for a tool, cheapest and most mechanical first.** Coverage before
mutation: coverage grades what no test *reaches*, mutation only grades the
tests that exist, so mutation on an unreached line tells you nothing while
looking productive. Then fuzz anything parsing untrusted input, and mutate
last, where judgement is actually needed. Learned the slow way here - the
pipeline's entire write path (`update_labels`, `add_comment`, `set_property`)
turned out never to have executed in a test, and a five-second coverage run
found it after a lot of careful mutation had not.

**The failure mode this code has, repeatedly.** Not crashes - a path reporting
success without the evidence that success implies. A gate passing on a probe
that tested nothing; a paging helper answering "no opt-out" from history it
never fetched; `ANTHROPIC_API_KEY set` proving only that a variable exists. A
green suite is weak evidence against this class, so when you add a guard, ask
what it would do if the thing it checks were unavailable rather than false.

## Layout

```
config.toml           pre-registered pilot config (cohort, labels, thresholds)
prompt/system.md      the rubric prompt (versioned)
docs/                 maintainer announcement (label names pinned by the tests)
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

