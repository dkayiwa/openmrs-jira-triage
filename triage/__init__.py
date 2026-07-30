"""O3 AI triage pilot: a deterministic pipeline around one Claude call per ticket.

The model only returns a classification object; every Jira read and write
happens in this code, so the pilot's guarantees (visible information only,
labels + comments only, opt-out on label removal) are properties of the code
and the bot's permission scheme, not of the prompt.

"Deterministic" describes the pipeline, NOT the classification. Measured by
evals/injection_eval.py: the same ticket text classified five times returned
automation_candidate four times and needs_judgment once. `temperature` is
deprecated on this model, so that spread cannot be pinned away in config. Two
consequences worth carrying:

- A ticket is only re-classified when its content hash or the prompt version
  changes, but when it is, the label can move for no reason but the draw - and
  a moved label posts a fresh comment to every watcher, then counts against the
  pre-registered label-removal kill metric if a maintainer prefers the first.
- The eval gate is >= 90% grader agreement, and a model that disagrees with
  itself on borderline tickets spends part of that budget on its own variance.

Neither is a bug to fix here; both are pilot-design calls for the owners, and
they need the measured number in front of them to make either.
"""
