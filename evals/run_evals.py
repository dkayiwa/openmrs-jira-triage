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
  python evals/run_evals.py [--min-agreement 0.9]
"""
from __future__ import annotations

import argparse
import csv
import pathlib
import re
import shutil
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
            if not re.fullmatch(r"[A-Z][A-Z0-9]*-\d+", (r.get("key") or "").strip()):
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
            graded_hash = (r.get("content_hash") or "").strip()
            if graded_hash and content_hash(src.read_text()) != graded_hash:
                print(f"skip {r['key']}: {src} changed since grading; re-run the "
                      "dry-run for this ticket and re-grade it")
                continue
            CONTEXTS.mkdir(exist_ok=True)
            shutil.copy(src, CONTEXTS / src.name)
            # The hash is recorded, not merely checked here: freezing is the
            # whole point of evals/contexts/, and without a stored hash a later
            # edit to a frozen file silently changes what the gate measures.
            rows[r["key"]] = {"key": r["key"], "expected_label": expected,
                              "content_hash": content_hash(src.read_text()),
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
    cases, rejected = [], []
    for row in rows:
        key = (row.get("key") or "").strip()
        expected = (row.get("expected_label") or "").strip()
        if not re.fullmatch(r"[A-Z][A-Z0-9]*-\d+", key):
            rejected.append(f"{key!r}: not a Jira issue key")
            continue
        if expected not in LABEL_KEYS:
            rejected.append(f"{key}: expected_label {expected!r} is not one of {LABEL_KEYS}")
            continue
        frozen = CONTEXTS / f"{key}.txt"
        if not frozen.is_file():
            rejected.append(f"{key}: frozen context {frozen} is missing")
            continue
        graded_hash = (row.get("content_hash") or "").strip()
        if not graded_hash:
            print(f"WARN {key}: no content_hash recorded, so the frozen context "
                  "cannot be verified against what was graded")
        elif content_hash(frozen.read_text()) != graded_hash:
            rejected.append(f"{key}: frozen context has been edited since grading")
            continue
        cases.append({"key": key, "expected_label": expected, "context": frozen})
    for reason in rejected:
        print(f"REJECT {reason}")
    if rejected:
        sys.exit(f"{len(rejected)} graded case(s) are unusable; the gate is not "
                 "meaningful until they are fixed or removed")
    return cases


def run(min_agreement: float) -> int:
    cases = load_cases()
    if not cases:
        sys.exit("no graded cases yet - fill in a proposals CSV and --import-proposals it first")
    cfg = load_config()
    clf = Classifier(cfg["claude"]["model"], cfg["claude"]["max_tokens"],
                     (ROOT / "prompt" / "system.md").read_text())
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
    agreement = hits / len(cases)
    # The model is named alongside the prompt: this gate is pre-registered
    # against a specific pair, and a result that records only one of them cannot
    # be reproduced later.
    print(f"\nagreement: {hits}/{len(cases)} = {agreement:.1%} "
          f"(model {cfg['claude']['model']}, prompt {cfg['prompt']['version']}, "
          f"gate {min_agreement:.0%})")
    for key, want, got_label in misses:
        print(f"  confusion: {key} {want} -> {got_label}")
    return 0 if agreement >= min_agreement else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--import-proposals", metavar="CSV", help="import grades from a filled proposals CSV")
    ap.add_argument("--contexts", default=str(ROOT / "out" / "contexts"),
                    help="where the proposal contexts live (default: out/contexts)")
    ap.add_argument("--min-agreement", type=float, default=0.9)
    args = ap.parse_args()
    if args.import_proposals:
        import_proposals(args.import_proposals, args.contexts)
        return 0
    return run(args.min_agreement)


if __name__ == "__main__":
    sys.exit(main())
