#!/usr/bin/env python3
"""Prompt regression harness.

evals/graded.csv holds human-graded expected labels; evals/contexts/ holds the
exact context each grade was made against (frozen, so eval results don't drift
as tickets change). Import grades from a filled-in proposals CSV, then run the
set after every prompt change. During the pilot, every human label-removal
should become a new case here - that closes the loop between the
"fewer than 1 in 10 removed" metric and prompt improvement.

Usage:
  python evals/run_evals.py --import-proposals out/proposals-XXXX.csv
  python evals/run_evals.py [--min-agreement 0.9]          # the gate: pinned model
  python evals/run_evals.py --model claude-sonnet-5 ...     # a labelled comparison
"""
from __future__ import annotations

import argparse
import csv
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from triage.classifier import LABEL_KEYS, Classifier  # noqa: E402
from triage.context import content_hash  # noqa: E402
from triage.run import load_config  # noqa: E402

EVALS = ROOT / "evals"
GRADED = EVALS / "graded.csv"
CONTEXTS = EVALS / "contexts"

GRADED_COLUMNS = ["key", "expected_label", "content_hash", "notes"]

# A security boundary, not a formatting check: the key becomes a filesystem
# path below. Defined once so a tightening cannot land on only one caller.
KEY_RE = re.compile(r"[A-Z][A-Z0-9]*-\d+")


def import_proposals(path: str, contexts_dir: str) -> None:
    rows: dict[str, dict] = {}
    if GRADED.exists():
        with open(GRADED) as fh:
            rows = {r["key"]: r for r in csv.DictReader(fh)}
    added = 0
    with open(path) as fh:
        for r in csv.DictReader(fh):
            grade = (r.get("grade(ok/wrong)") or "").strip().lower()
            if grade not in ("ok", "wrong"):
                continue
            # The key becomes a filesystem path below and is stored for later
            # reads, so it is bounded to a Jira key shape rather than trusted.
            if not KEY_RE.fullmatch((r.get("key") or "").strip()):
                print(f"skip {r.get('key')!r}: not a Jira issue key")
                continue
            # An `ok` grade copies the proposal's own label into expected_label.
            # If that proposal came from a replayed file rather than the pinned
            # model, the eval set would then measure the pinned model against a
            # different classifier's decision boundary - and the >= 90% gate it
            # feeds is what authorises go-live. A `wrong` grade is safe either
            # way, because correct_label is the human's judgement.
            # Fails CLOSED on an absent column: this gate authorises go-live, and
            # a sheet with no `source` at all (hand-made, or round-tripped
            # through a spreadsheet that dropped the column) is unknown
            # provenance, not proven-api provenance.
            source = (r.get("source") or "").strip()
            if grade == "ok" and source != "api":
                stated = f"source={source}" if source else "no source column"
                print(f"skip {r['key']}: {stated}, so an 'ok' grade would seed the eval "
                      "set with an unverified classifier's own label; re-run the dry-run "
                      "for this ticket through the API path, or mark it wrong with an "
                      "explicit correct_label")
                continue
            expected = r["proposed_label"] if grade == "ok" else (r.get("correct_label") or "").strip()
            if expected not in LABEL_KEYS:
                print(f"skip {r['key']}: correct_label must be one of {LABEL_KEYS}")
                continue
            src = pathlib.Path(contexts_dir) / (r["key"] + ".txt")
            if not src.exists():
                print(f"skip {r['key']}: no context at {src}")
                continue
            # out/contexts/<KEY>.txt is overwritten by every dry-run, so a later
            # run can replace the text this grade was made against. Freezing the
            # new context against the old grade would silently corrupt the set
            # that gates the >= 90% go-live decision.
            # Fails CLOSED on an absent hash, like the source gate above: with
            # no hash there is nothing to prove the context on disk is the text
            # that was graded, and freezing the wrong text corrupts the set that
            # authorises go-live.
            graded_hash = (r.get("content_hash") or "").strip()
            if not graded_hash:
                print(f"skip {r['key']}: no content_hash, so the sheet cannot show "
                      "which text was graded; re-export the proposals CSV and "
                      "re-grade from it")
                continue
            # Read ONCE, then verify and freeze that exact text. This used to
            # read src three times - to check, to copy, and to hash for storage
            # - and out/contexts/<KEY>.txt is overwritten by every dry-run. A
            # sweep landing between those reads left a frozen file and a stored
            # hash that agreed with each other and with nothing a human had
            # graded, while "imported N" reported a verified freeze.
            text = src.read_text()
            if content_hash(text) != graded_hash:
                print(f"skip {r['key']}: {src} changed since grading; re-run the "
                      "dry-run for this ticket and re-grade it")
                continue
            CONTEXTS.mkdir(exist_ok=True)
            (CONTEXTS / src.name).write_text(text)
            # The stored hash is the one that was verified, not a fresh read.
            rows[r["key"]] = {"key": r["key"], "expected_label": expected,
                              "content_hash": graded_hash,
                              "notes": r.get("grader_notes", "")}
            added += 1
    with open(GRADED, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=GRADED_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for row in sorted(rows.values(), key=lambda x: x["key"]):
            w.writerow(row)
    print(f"imported {added} graded case(s); eval set now has {len(rows)}")


def load_cases() -> list[dict]:
    """Graded cases whose frozen context still matches what was graded.

    A case that cannot be trusted is refused rather than scored: this set gates
    go-live, so silently measuring the prompt against an edited context or a
    corrupted label would move the gate without anyone seeing it.
    """
    if not GRADED.exists():
        sys.exit(f"{GRADED} does not exist - --import-proposals a graded sheet first")
    with open(GRADED) as fh:
        rows = list(csv.DictReader(fh))
    cases, rejected, seen = [], [], set()
    for row in rows:
        key = (row.get("key") or "").strip()
        expected = (row.get("expected_label") or "").strip()
        if not KEY_RE.fullmatch(key):
            rejected.append(f"{key!r}: not a Jira issue key")
            continue
        # graded.csv is checked in and hand-edited. A repeated key would
        # weight one human judgement twice in the gate's denominator, and
        # silently - the agreement percentage would still look ordinary.
        if key in seen:
            rejected.append(f"{key}: appears more than once, so one grade would "
                            "count twice toward the gate")
            continue
        seen.add(key)
        if expected not in LABEL_KEYS:
            rejected.append(f"{key}: expected_label {expected!r} is not one of {LABEL_KEYS}")
            continue
        frozen = CONTEXTS / f"{key}.txt"
        if not frozen.is_file():
            rejected.append(f"{key}: frozen context {frozen} is missing")
            continue
        graded_hash = (row.get("content_hash") or "").strip()
        if not graded_hash:
            # Refused, not warned: the docstring above promises exactly this,
            # and a case whose frozen text cannot be verified was still being
            # scored into the gate that authorises go-live.
            rejected.append(f"{key}: no content_hash recorded, so the frozen "
                            "context cannot be verified against what was graded")
            continue
        if content_hash(frozen.read_text()) != graded_hash:
            rejected.append(f"{key}: frozen context has been edited since grading")
            continue
        cases.append({"key": key, "expected_label": expected, "context": frozen})
    for reason in rejected:
        print(f"REJECT {reason}")
    if rejected:
        sys.exit(f"{len(rejected)} graded case(s) are unusable; the gate is not "
                 "meaningful until they are fixed or removed")
    return cases


def score(model: str, max_tokens: int, prompt: str, cases: list[dict]) -> tuple[int, list]:
    """Run every graded case against one model. Returns (hits, misses)."""
    clf = Classifier(model, max_tokens, prompt)
    hits = 0
    misses: list[tuple[str, str, str]] = []
    for case in cases:
        # Per-case isolation: one API hiccup must not discard the paid
        # classifications already made; an errored case counts as a miss.
        try:
            got_label = clf.classify(case["context"].read_text()).label
        except Exception as e:
            got_label = f"ERROR ({type(e).__name__}: {str(e)[:120]})"
        if got_label == case["expected_label"]:
            hits += 1
            print("  " + case["key"] + ": ok")
        else:
            misses.append((case["key"], case["expected_label"], got_label))
            print("  " + case["key"] + ": MISS expected " + case["expected_label"]
                  + " got " + got_label)
    return hits, misses


def run(min_agreement: float, models: list[str] | None = None) -> int:
    """Score the graded set. With no `models`, this IS the pre-registered gate.

    `models` runs a comparison instead, and is deliberately additive: there is no
    way to make the *gate* measure a model other than the one config.toml pins.
    The pinned pair is what the pilot committed to before seeing results, so a
    model shopped for afterwards must not be able to inherit its authority - the
    comparison says so in its own output, and its exit code reports whether the
    comparison ran, not whether anything was cleared for go-live.
    """
    cases = load_cases()
    if not cases:
        sys.exit("no graded cases yet - fill in a proposals CSV and --import-proposals it first")
    cfg = load_config()
    prompt = (ROOT / "prompt" / "system.md").read_text()
    pinned, max_tokens = cfg["claude"]["model"], cfg["claude"]["max_tokens"]

    if not models:
        hits, misses = score(pinned, max_tokens, prompt, cases)
        agreement = hits / len(cases)
        # The model is named alongside the prompt: this gate is pre-registered
        # against a specific pair, and a result that records only one of them
        # cannot be reproduced later.
        print(f"\nagreement: {hits}/{len(cases)} = {agreement:.1%} "
              f"(model {pinned}, prompt {cfg['prompt']['version']}, "
              f"gate {min_agreement:.0%})")
        for key, want, got_label in misses:
            print(f"  confusion: {key} {want} -> {got_label}")
        return 0 if agreement >= min_agreement else 1

    print(f"COMPARISON - not the pre-registered gate. config.toml pins {pinned}, and "
          f"only that model's result authorises go-live. Nothing here changes what "
          f"the pilot committed to, and the exit code below reports whether the "
          f"comparison ran - not whether any model was cleared.\n")
    results = []
    for model in models:
        print(f"{model}:")
        hits, misses = score(model, max_tokens, prompt, cases)
        # Errors deflate agreement exactly like disagreement does, so a model
        # that could not be reached reads as "0/12 below" - identical in the
        # table to one that answered and was wrong every time.
        errored = sum(1 for _, _, got in misses if got.startswith("ERROR"))
        results.append((model, hits, hits / len(cases), errored))
        for key, want, got_label in misses:
            print(f"  confusion: {key} {want} -> {got_label}")
        print()
    width = max(len(m) for m, *_ in results)
    print(f"{'model'.ljust(width)}  agreement        vs {min_agreement:.0%}")
    for model, hits, agreement, errored in results:
        # Deliberately not PASS/FAIL: that vocabulary belongs to the gate, and a
        # comparison row reading PASS is exactly the misreading this guards.
        standing = "at or above" if agreement >= min_agreement else "below"
        note = "  (pinned)" if model == pinned else ""
        if errored:
            note += f"  [{errored} of {len(cases)} errored - not disagreement]"
        print(f"{model.ljust(width)}  {hits}/{len(cases)} {agreement:6.1%}  "
              f"{standing}{note}")
    if pinned not in models:
        print(f"\n{pinned} (pinned) was not run; add --model {pinned} to include it, "
              "or run with no --model at all for the gate itself.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--import-proposals", metavar="CSV", help="import grades from a filled proposals CSV")
    ap.add_argument("--contexts", default=str(ROOT / "out" / "contexts"),
                    help="where the proposal contexts live (default: out/contexts)")
    ap.add_argument("--min-agreement", type=float, default=0.9)
    ap.add_argument("--model", action="append", metavar="ID",
                    help="score this model instead of the pinned one, repeatable. "
                         "Runs a labelled COMPARISON, never the gate; config.toml is "
                         "not modified and the pinned model still decides go-live")
    args = ap.parse_args()
    if args.import_proposals:
        import_proposals(args.import_proposals, args.contexts)
        return 0
    return run(args.min_agreement, args.model)


if __name__ == "__main__":
    sys.exit(main())
