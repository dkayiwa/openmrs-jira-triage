"""Fuzz wiki_safe/csv_safe against their stated invariants.

    python -m evals.fuzz_sanitisers      # offline, no credential, a few seconds

Kept in the repo because the README claims the sanitisers are fuzzed, and a
claim whose evidence lives on one laptop is the thing this codebase keeps
getting wrong. The findings became unit tests; this is what re-runs the
whole input space after a change to either function.

Both have only ever been tested against payloads someone thought of. wiki_safe
is a security boundary: its output goes into a public Jira comment, and its
docstring makes structural claims (backslashes stripped first, mentions and
images disabled, newlines collapsed) that spot-checks cannot establish over the
whole input space.
"""
import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from triage.run import csv_safe, wiki_safe

random.seed(20260731)

ALPHA = list("[]!<>\\/{}~*_-+=@?#|&\"'`^%$ abAB09\n\r\t") + [
    " ",      # no-break space
    " ",      # line separator
    " ",      # paragraph separator
    "",      # next line
    "​",      # zero-width space
    "﻿",      # BOM / zero-width no-break
    "‮",      # right-to-left override
    "‏",      # right-to-left mark
    "{{", "}}", "[~accountid:712020:abc]", "!https://x/p.png!", "\\\\",
]

fails = {}


def note(key, sample):
    fails.setdefault(key, sample)


for _ in range(60000):
    s = "".join(random.choice(ALPHA) for _ in range(random.randint(0, 24)))
    out = wiki_safe(s)

    if any(c in out for c in "\n\r\t"):
        note("newline or tab survives", (s, out))
    for j, c in enumerate(out):
        if c in "[!<" and (j == 0 or out[j - 1] != "\\"):
            note("unescaped special character", (s, out))
            break
        if c == "\\" and (j + 1 >= len(out) or out[j + 1] not in "[!<"):
            note("stray backslash", (s, out))
            break
    if wiki_safe(out) != out:
        note("not idempotent", (s, out))
    # Characters that reorder or hide text in a rendered comment.
    for bad, label in (("‮", "bidi override"), ("‏", "RTL mark"),
                       ("​", "zero-width space"), ("﻿", "BOM")):
        if bad in out:
            note(f"{label} reaches the comment", (s, out))

    c = csv_safe(s)
    if c[:1] in ("=", "+", "-", "@", "\t", "\r", "\n"):
        note("csv formula prefix survives", (s, c))

print(f"60000 random inputs; {len(fails)} invariant(s) violated")
for key, (s, out) in fails.items():
    print(f"  {key}\n    in : {s!r}\n    out: {out!r}")

sys.exit(1 if fails else 0)
