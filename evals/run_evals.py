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
            rows[r["key"]] = {"key": r["key"], "expected_label": expected,
                              "notes": r.get("grader_notes", "")}
            added += 1
    with open(GRADED, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["key", "expected_label", "notes"])
        w.writeheader()
        for row in sorted(rows.values(), key=lambda x: x["key"]):
            w.writerow(row)
    print(f"imported {added} graded case(s); eval set now has {len(rows)}")


def run(min_agreement: float) -> int:
    with open(GRADED) as fh:
        cases = list(csv.DictReader(fh))
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
            got_label = clf.classify((CONTEXTS / (case["key"] + ".txt")).read_text()).label
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
    print(f"\nagreement: {hits}/{len(cases)} = {agreement:.0%} "
          f"(prompt {cfg['prompt']['version']}, gate {min_agreement:.0%})")
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
