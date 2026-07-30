"""Offline tests for the pure pipeline logic (no network, no API key).

Fixture shapes mirror real Jira Cloud v2 responses: changelog label lists,
paginated changelog envelopes and missing-property status codes were all
verified live against openmrs.atlassian.net. Classifier response handling is
covered here against a stub client; only the real HTTP request needs
credentials, and that is exercised by evals/run_evals.py.
"""
from __future__ import annotations

import collections
import contextlib
import csv
import datetime
import html
import io
import itertools
import json
import os
import re
import sys
import tempfile
import types
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from triage import context as ctx  # noqa: E402
from triage import github as gh  # noqa: E402
from triage import metrics, preflight, run  # noqa: E402
from triage.classifier import (  # noqa: E402
    LABEL_KEYS,
    SCHEMA,
    Classification,
    Classifier,
    clamp_classification,
    validate_classification,
)
from triage.jira import JiraClient, JiraError  # noqa: E402
from triage.metrics import decide, parse_launch, sla_met  # noqa: E402
from triage.run import (  # noqa: E402
    _load_dotenv,
    bot_identity_error,
    comment_body,
    load_config,
    plan_label_writes,
    plan_ticket,
)
from triage.state import TicketState, inspect  # noqa: E402

# Read from config rather than hardcoded: a legitimate prompt-version bump is a
# normal event, and these fixtures assert behaviour, not a particular string.
PROMPT_VERSION = load_config()["prompt"]["version"]

AI = ["ai-triage-automation-candidate", "ai-triage-needs-judgment", "ai-triage-needs-more-info"]


class StubGitHub:
    """Offline stand-in for GitHubClient, installed over every entry point.

    The open-PR backstop is enabled in config.toml, so without this every test
    that calls run.main() or preflight.main() would issue real searches - and
    the unauthenticated throttle is 6s each, which turned the suite from half a
    second into hours. State is class-level because the client is constructed
    inside main(), out of the test's reach.
    """

    answers: dict = {}
    searched: list = []
    error: Exception | None = None
    built: int = 0

    @classmethod
    def reset(cls, answers=None, error=None):
        cls.answers = dict(answers or {})
        cls.searched = []
        cls.error = error
        cls.built = 0

    def __init__(self, org, token=None, **kwargs):
        self.org = org
        self.authenticated = bool(token)
        self.min_interval = 0.0
        type(self).built += 1

    def open_pr_urls(self, key):
        type(self).searched.append(key)
        if type(self).error:
            raise type(self).error
        return list(type(self).answers.get(key, []))


_GITHUB_PATCHES = []


class StubAnthropicModels:
    """Stands in for the credential probe so no test reaches the API.

    Same lesson as StubGitHub above, relearned: adding a live probe to
    preflight without a suite-wide stub sent every preflight test to the real
    endpoint. The suite went from 2.8s to 17.2s, which is the only reason it
    was noticed - a slower probe would just have been an unexplained cost on
    every CI run, and a flaky one an unexplained failure.
    """

    available = [load_config()["claude"]["model"], "claude-haiku-4-5"]
    error: Exception | None = None

    def __init__(self, *a, **k):
        self.models = self

    def list(self, limit=100):
        if type(self).error:
            raise type(self).error
        return types.SimpleNamespace(
            data=[types.SimpleNamespace(id=m) for m in type(self).available])

    @classmethod
    def reset(cls):
        cls.available = [load_config()["claude"]["model"], "claude-haiku-4-5"]
        cls.error = None


def setUpModule():
    # One patch point: preflight builds its client through run.github_from_env,
    # so both entry points resolve GitHubClient in run's namespace.
    StubGitHub.reset()
    patch = mock.patch.object(run, "GitHubClient", StubGitHub)
    patch.start()
    _GITHUB_PATCHES.append(patch)
    StubAnthropicModels.reset()
    patch = mock.patch.object(preflight, "Anthropic", StubAnthropicModels)
    patch.start()
    _GITHUB_PATCHES.append(patch)


def tearDownModule():
    while _GITHUB_PATCHES:
        _GITHUB_PATCHES.pop().stop()


def issue(labels=(), histories=(), comments=(), **fields):
    base = {
        "summary": "Fix the widget", "description": "It is broken.",
        "labels": list(labels), "issuelinks": [], "parent": None,
        "status": {"name": "To Do"},
        "comment": {"comments": list(comments), "total": len(comments)},
    }
    base.update(fields)
    return {"key": "O3-1", "fields": base, "changelog": {"histories": list(histories)}}


def label_change(author_id, frm, to, display="Someone", created="2026-07-29T10:00:00.000+0000"):
    return {
        "author": {"accountId": author_id, "displayName": display},
        "created": created,
        "items": [{"field": "labels", "fromString": frm, "toString": to}],
    }


class StubJira:
    """Stands in for JiraClient in context assembly (parent fetch, comments)."""

    def issue(self, key, fields, expand_changelog=False):
        return {"key": key, "fields": {"summary": "Parent summary", "description": "Parent desc"}}

    def comments(self, key, embedded):
        return list((embedded or {}).get("comments", []))


class StubResponse:
    text = "{}"
    request = type("R", (), {"method": "GET", "url": "stub"})()

    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def json(self):
        return self._payload


class StubSession:
    """Replays queued responses and records what was requested."""

    def __init__(self, pages):
        self.headers: dict = {}
        self.pages = list(pages)
        self.calls: list[dict] = []

    def get(self, url, params=None, timeout=None):
        assert timeout is not None, "every network call must carry a timeout"
        self.calls.append({"method": "GET", "url": url, **(params or {})})
        return self.pages.pop(0)

    def post(self, url, json=None, timeout=None):
        assert timeout is not None, "every network call must carry a timeout"
        self.calls.append({"method": "POST", "url": url, **(json or {})})
        return self.pages.pop(0)

    def put(self, url, params=None, json=None, timeout=None):
        assert timeout is not None, "every network call must carry a timeout"
        self.calls.append({"method": "PUT", "url": url, "params": params or {},
                           "body": json})
        return self.pages.pop(0)

    def delete(self, url, timeout=None):
        assert timeout is not None, "every network call must carry a timeout"
        self.calls.append({"method": "DELETE", "url": url})
        return self.pages.pop(0)


class StubClassifier:
    def __init__(self, classification):
        self.classification = classification

    def classify(self, text):
        return self.classification


class RecordingJira:
    """Stub JiraClient recording every write, for the live-mode run tests."""

    authenticated = True

    def __init__(self, issues):
        self.issues = issues
        self.properties: dict = {}
        self.writes: list[tuple] = []

    def myself(self):
        return {"accountId": "bot", "displayName": "Triage Bot"}

    def fields(self):
        return []

    def search_keys(self, jql):
        return list(self.issues)

    def issue(self, key, fields, expand_changelog=False):
        return self.issues[key]

    def changelog(self, key, embedded):
        return list((embedded or {}).get("histories", []))

    def comments(self, key, embedded):
        return list((embedded or {}).get("comments", []))

    def get_property(self, key, prop):
        return self.properties.get(key)

    def set_property(self, key, prop, value):
        self.properties[key] = value
        self.writes.append(("property", key))

    def update_labels(self, key, add, remove):
        self.writes.append(("labels", key, tuple(add), tuple(remove)))
        # Applied, not just recorded: a stub that accepted writes without
        # reflecting them would let a second run see pre-write state, which is
        # exactly the kind of dishonest double that hides a real regression.
        current = self.issues[key]["fields"]["labels"]
        self.issues[key]["fields"]["labels"] = (
            [l for l in current if l not in remove] + list(add)
        )

    def add_comment(self, key, body):
        self.writes.append(("comment", key, body))


class LivingJira(RecordingJira):
    """RecordingJira, but the changelog reflects what was written.

    The parent already applies label writes to the labels field. It does not
    record them in the changelog, so no test had ever seen the bot read back
    its OWN add on a later sweep - and that add is half of the pair the opt-out
    guarantee is decided from. A stub that accepts a write and then denies it
    happened cannot show a sequence bug, only confirm the fixture.
    """

    def __init__(self, issues):
        super().__init__(issues)
        self.clock = 0

    def _stamp(self) -> str:
        self.clock += 1
        return f"2026-08-10T{9 + self.clock:02d}:00:00.000+0000"

    def _log(self, key, author, display, frm, to):
        self.issues[key]["changelog"]["histories"].append(
            label_change(author, frm, to, display=display, created=self._stamp()))

    def update_labels(self, key, add, remove):
        before = " ".join(self.issues[key]["fields"]["labels"])
        super().update_labels(key, add, remove)
        after = " ".join(self.issues[key]["fields"]["labels"])
        self._log(key, "bot", "Triage Bot", before, after)

    def human_removes(self, key, label):
        """A maintainer takes the label off - the pilot's opt-out gesture."""
        before = " ".join(self.issues[key]["fields"]["labels"])
        self.issues[key]["fields"]["labels"] = [
            l for l in self.issues[key]["fields"]["labels"] if l != label]
        after = " ".join(self.issues[key]["fields"]["labels"])
        self._log(key, "maintainer", "A Maintainer", before, after)


class StateTests(unittest.TestCase):
    def test_human_removal_is_opt_out(self):
        st = inspect(issue(histories=[label_change("u1", AI[2], "")]), AI, "bot")
        self.assertTrue(st.opted_out)
        self.assertEqual(st.opted_out_by, "Someone")

    def test_bot_removal_is_not_opt_out(self):
        st = inspect(issue(histories=[label_change("bot", AI[2], "")]), AI, "bot")
        self.assertFalse(st.opted_out)

    def test_unknown_bot_id_treats_any_removal_as_opt_out(self):
        st = inspect(issue(histories=[label_change("bot", AI[2], "")]), AI, None)
        self.assertTrue(st.opted_out)

    def test_space_and_comma_separated_label_lists(self):
        st = inspect(
            issue(histories=[label_change("u1", "intro " + AI[0], "intro"),
                             label_change("u2", "x," + AI[1], "x")]),
            AI, "bot",
        )
        self.assertTrue(st.opted_out)

    def test_non_ai_label_removal_is_ignored(self):
        st = inspect(issue(histories=[label_change("u1", "intro", "")]), AI, "bot")
        self.assertFalse(st.opted_out)

    def test_human_add_flagged_as_violation(self):
        st = inspect(issue(histories=[label_change("u1", "", AI[0])]), AI, "bot")
        self.assertEqual(st.human_adds, ["Someone"])

    def test_ai_labels_present_filtered(self):
        st = inspect(issue(labels=[AI[0], "intro"]), AI, "bot")
        self.assertEqual(st.ai_labels_present, [AI[0]])

    def test_truncated_embedded_changelog_raises_rather_than_truncating(self):
        # The convenience default must not become a quiet correctness hole: a
        # truncated page can spot an opt-out but cannot find the bot's first add.
        iss = issue()
        iss["changelog"] = {"histories": [{"items": []}] * 100, "total": 264}
        with self.assertRaises(ValueError) as caught:
            inspect(iss, AI, "bot")
        self.assertIn("truncated", str(caught.exception))

    def test_complete_embedded_changelog_is_accepted(self):
        iss = issue(histories=[label_change("bot", "", AI[0])])
        iss["changelog"]["total"] = 1
        self.assertIsNone(inspect(iss, AI, "bot").opted_out_by)

    def test_bot_first_label_time_captured(self):
        st = inspect(
            issue(histories=[
                label_change("bot", "", AI[2], created="2026-08-01T09:00:00.000+0000"),
                label_change("bot", AI[2], AI[0], created="2026-08-02T09:00:00.000+0000"),
            ]),
            AI, "bot",
        )
        self.assertEqual(st.bot_first_labeled_at, "2026-08-01T09:00:00.000+0000")

    def test_human_add_does_not_set_bot_label_time(self):
        st = inspect(issue(histories=[label_change("u1", "", AI[0])]), AI, "bot")
        self.assertIsNone(st.bot_first_labeled_at)


class WorkflowInvariantTests(unittest.TestCase):
    """The file that decides whether a run writes to public tickets.

    It had no coverage at all. An edit here cannot be caught by any other test:
    nothing else determines when --live is passed, whether the suite gates the
    sweep, or whether a dispatch input reaches a shell.
    """

    @classmethod
    def setUpClass(cls):
        import yaml

        path = Path(__file__).resolve().parent.parent / ".github/workflows/triage.yml"
        cls.text = path.read_text()
        cls.doc = yaml.safe_load(cls.text)
        cls.job = cls.doc["jobs"]["triage"]
        cls.steps = cls.job["steps"]
        cls.sweep = next(s for s in cls.steps if "triage.run" in str(s.get("run", "")))

    def test_the_token_is_read_only(self):
        # The bot authenticates to Jira with its own secret; the workflow token
        # needs nothing but checkout.
        self.assertEqual(self.doc["permissions"], {"contents": "read"})

    def test_dispatch_input_never_reaches_a_shell(self):
        # A workflow_dispatch input is arbitrary text. Interpolating it into a
        # run: line executes it; it must arrive as an environment variable.
        self.assertNotIn("${{ inputs.", self.sweep["run"],
                         "a dispatch input is interpolated into the shell line")
        self.assertIn("$LIMIT", self.sweep["run"])
        self.assertIn("LIMIT", self.sweep["env"])

    def test_live_is_only_reachable_deliberately(self):
        # --live must appear exactly once, guarded by the schedule or an explicit
        # mode choice. Anything else risks unintended writes to public tickets.
        run_line = self.sweep["run"]
        self.assertEqual(run_line.count("--live"), 1)
        guard = run_line.split("--live")[0]
        self.assertIn("github.event_name == 'schedule'", guard)
        self.assertIn("inputs.mode == 'live'", guard)

    def test_a_scheduled_run_is_live_if_the_schedule_is_ever_enabled(self):
        # The schedule block is commented out until the eval gate passes; when it
        # is enabled, scheduled runs must already be the live path.
        if "schedule" in (self.doc.get(True) or self.doc.get("on") or {}):
            self.assertIn("github.event_name == 'schedule'", self.sweep["run"])
        self.assertIn("cron", self.text, "the schedule block should remain documented")

    def test_the_suite_gates_the_sweep(self):
        # If the tests ran after the sweep, or not at all, every guard in this
        # file would be advisory.
        indexes = {"tests": None, "sweep": None}
        for i, step in enumerate(self.steps):
            run_line = str(step.get("run", ""))
            if "unittest" in run_line:
                indexes["tests"] = i
            if "triage.run" in run_line:
                indexes["sweep"] = i
        self.assertIsNotNone(indexes["tests"], "no step runs the test suite")
        self.assertLess(indexes["tests"], indexes["sweep"],
                        "the suite must run before the sweep")

    def test_artifacts_survive_a_failed_run(self):
        # A failed sweep is exactly when the journal is needed, and the run now
        # exits non-zero whenever any ticket errored.
        upload = next(s for s in self.steps if "upload-artifact" in str(s.get("uses", "")))
        self.assertEqual(upload.get("if"), "always()")
        self.assertIn("out/", str(upload["with"]["path"]))

    def test_sweeps_cannot_overlap(self):
        # Two concurrent sweeps could both read a ticket before either labelled
        # it, and each would post a comment to every watcher.
        self.assertIn("group", self.doc["concurrency"])
        self.assertIs(self.doc["concurrency"].get("cancel-in-progress"), False)

    def test_no_secret_is_interpolated_into_a_shell_line(self):
        for step in self.steps:
            self.assertNotIn("secrets.", str(step.get("run", "")),
                             f"secret interpolated into a run: line: {step}")

    def test_every_credential_the_pipeline_reads_is_supplied(self):
        # A missing secret degrades silently: the sweep would run anonymously and
        # skip every ticket, or fail every ticket, on a schedule.
        for name in ("JIRA_EMAIL", "JIRA_API_TOKEN", "TRIAGE_BOT_ACCOUNT_ID",
                     "ANTHROPIC_API_KEY", "GITHUB_TOKEN"):
            self.assertIn(name, self.sweep["env"], f"{name} is not passed to the sweep")

    def test_the_job_cannot_hold_the_concurrency_group_indefinitely(self):
        # This invariant only exists because of the one above. Never cancelling
        # a running sweep is correct - it could be mid-write - but it means a
        # wedged run owns the group, and GitHub keeps just one pending run per
        # group and cancels the rest. On the default 6h timeout a single hang
        # swallows the next 4-hourly sweeps, and the 24h SLA is a pre-registered
        # metric, so it is missed with nothing reporting it.
        timeout = self.job.get("timeout-minutes")
        self.assertIsNotNone(timeout, "the job has no timeout-minutes")
        self.assertLess(timeout, 4 * 60, "a timeout longer than the cron interval "
                                         "cannot stop runs from queueing behind a hang")

    def test_the_gate_runs_with_the_credentials_the_sweep_will_use(self):
        # Running preflight only from a laptop proves the pipeline against a
        # personal token. The backstop's production credential is github.token -
        # repo-scoped, ephemeral, and searching a different org - and if it
        # cannot do that, every ticket fails the backstop and the breaker aborts
        # the first live sweep having written nothing.
        idx = {}
        for i, step in enumerate(self.steps):
            line = str(step.get("run", ""))
            if "triage.preflight" in line:
                idx["gate"] = i
            if "triage.run" in line:
                idx["sweep"] = i
        self.assertIn("gate", idx, "no step runs the preflight gate")
        self.assertLess(idx["gate"], idx["sweep"], "the gate must run before the sweep")
        gate_env = self.steps[idx["gate"]]["env"]
        self.assertIn("github.token", gate_env["GITHUB_TOKEN"],
                      "the gate must probe GitHub with the sweep's own token")
        # Every credential the sweep gets, so the gate's report describes the
        # sweep's environment rather than a subset of it.
        for name in self.sweep["env"]:
            if name == "LIMIT":
                continue
            self.assertIn(name, gate_env,
                          f"the sweep has {name} but the gate does not, so the gate "
                          "reports on an environment that is not the sweep's")

    def test_the_local_live_guard_reads_this_exact_file(self):
        # schedule_conflict() parses this file on every local --live run to
        # refuse a sweep that could race the scheduled one. Every test of it
        # uses synthetic YAML, so the coupling to the real file was unproven:
        # a workflow edit could silently make the guard blind, and the failure
        # only appears as two sweeps commenting on the same ticket. Checks both
        # states against the real text, so enabling the schedule cannot quietly
        # leave local --live unguarded.
        root = Path(__file__).resolve().parent.parent
        self.assertIsNone(run.schedule_conflict(root, in_ci=False),
                          "the schedule is commented out, so local --live is allowed")
        enabled = re.sub(r"^  # (schedule:|  - cron:)", r"  \1", self.text, flags=re.M)
        self.assertIn("\n  schedule:", enabled, "the commented block did not uncomment")
        with tempfile.TemporaryDirectory() as d:
            wf = Path(d) / ".github" / "workflows"
            wf.mkdir(parents=True)
            (wf / "triage.yml").write_text(enabled)
            self.assertIsNotNone(run.schedule_conflict(Path(d), in_ci=False),
                                 "with the schedule live, local --live must be refused")

    def test_the_open_pr_backstop_is_not_left_unauthenticated_on_a_schedule(self):
        # Unauthenticated search allows 10/min, so a scheduled full-cohort sweep
        # would spend minutes throttling and risk failing tickets on the limit.
        # github.token needs no new repo secret and no extra permission.
        self.assertIn("github.token", self.sweep["env"]["GITHUB_TOKEN"])


class DocumentedSurfaceTests(unittest.TestCase):
    """The README and config.toml against what the code actually offers.

    Both drift silently and in the direction that wastes someone else's time:
    a flag renamed in argparse leaves the README telling a maintainer to type
    something that exits 2, and a config key the code reads but the file lacks
    is a KeyError on a scheduled live run. These were audited by hand once;
    a hand audit that is not a test is a fact about one afternoon.
    """

    ROOT = Path(__file__).resolve().parent.parent
    MODULES = {"triage.run": "triage/run.py", "triage.preflight": "triage/preflight.py",
               "triage.metrics": "triage/metrics.py",
               "evals/run_evals.py": "evals/run_evals.py"}

    @classmethod
    def setUpClass(cls):
        cls.readme = (cls.ROOT / "README.md").read_text()

    def test_every_flag_the_readme_tells_you_to_type_is_accepted(self):
        accepted = {
            mod: set(re.findall(r'add_argument\(\s*"(--[a-z0-9-]+)"',
                                (self.ROOT / path).read_text()))
            for mod, path in self.MODULES.items()
        }
        # Any invocation, however it is prefixed: the README writes most of them
        # as `.venv/bin/python -m ...` inside fenced blocks and a few inline.
        # Anchoring on a line-initial "python" found two flags out of nineteen,
        # which the floor below caught - a passing extraction test that extracts
        # nothing is the vacuous-guard failure this whole suite keeps hunting.
        commands = [m.group(0).split("#")[0]
                    for m in re.finditer(r'python (?:-m triage\.[a-z_]+|evals/[a-z_]+\.py)'
                                         r'[^\n`]*', self.readme)]
        checked = 0
        for command in commands:
            mod = next((m for m in self.MODULES if m in command), None)
            if not mod:
                continue
            for flag in re.findall(r'(--[a-z0-9-]+)', command):
                checked += 1
                self.assertIn(flag, accepted[mod],
                              f"README says `{command.strip()}` but {mod} has no {flag}")
        self.assertGreater(checked, 5, "the extraction found almost no flags to check, "
                                       "so a passing result here means nothing")

    def test_every_flag_the_readme_mentions_in_prose_still_exists(self):
        # The prose list ("Useful flags: `--keys ...`, `--no-classify`") carries
        # no module context, so it is checked against the union - weaker than
        # the per-command check above, but it is what catches a rename in the
        # half of the README that is sentences rather than shell blocks, and
        # that half is where a reader looks first.
        accepted = set()
        for path in self.MODULES.values():
            accepted |= set(re.findall(r'add_argument\(\s*"(--[a-z0-9-]+)"',
                                       (self.ROOT / path).read_text()))
        mentioned = set(re.findall(r'`(--[a-z0-9-]+)[^`]*`', self.readme))
        self.assertGreater(len(mentioned), 3, "extraction found nothing to check")
        self.assertEqual(mentioned - accepted, set(),
                         "the README names flags no module accepts")

    def test_every_runnable_the_readme_names_exists(self):
        targets = set(re.findall(r'python -m ((?:triage|evals)\.[a-z_]+)', self.readme))
        targets |= set(re.findall(r'python (evals/[a-z_]+\.py)', self.readme))
        self.assertGreater(len(targets), 2, "extraction found nothing to check")
        for target in targets:
            path = target if target.endswith(".py") else target.replace(".", "/") + ".py"
            self.assertTrue((self.ROOT / path).exists(),
                            f"README tells you to run {target}, which does not exist")

    def test_every_config_key_the_code_reads_is_present(self):
        # Direction that matters: a key the code reads but the file lacks is a
        # KeyError mid-sweep. The reverse (an unused key) is only clutter, and
        # cannot be checked this way anyway - most sections are read through a
        # bound intermediate like `m = cfg["metrics"]`, which no regex sees.
        cfg = load_config()
        source = "\n".join((self.ROOT / p).read_text() for p in
                           ("triage/run.py", "triage/metrics.py", "triage/preflight.py",
                            "triage/context.py", "evals/run_evals.py"))
        pattern = r'cfg\[["\']([a-z_]+)["\']\]\[["\']([a-z_]+)["\']\]'
        found = set(re.findall(pattern, source))
        self.assertGreater(len(found), 5, "extraction found nothing to check")
        for section, key in sorted(found):
            self.assertIn(section, cfg, f'code reads cfg["{section}"], which is absent')
            self.assertIn(key, cfg[section],
                          f'code reads cfg["{section}"]["{key}"], which is absent')


class ReportMarkupTests(unittest.TestCase):
    """No untrusted field may introduce an element into the HTML report.

    The report is built by string concatenation and it is the document Dennis
    and Veronica open to review what the bot wrote. Reading the source for
    missing esc() calls is the weak check - 41 interpolations, and judging each
    by eye is how one gets missed. This puts a payload in every field that
    comes from a ticket or from the model, then parses the result and asks the
    parser which elements exist.
    """

    PAYLOAD = '<script>alert(1)</script><img src=x onerror=alert(2)>"><b>x</b>'

    def _elements(self, html_text):
        import html.parser

        found = []

        class Collect(html.parser.HTMLParser):
            def handle_starttag(self, tag, attrs):
                found.append((tag, dict(attrs)))

        Collect().feed(html_text)
        return found

    def _report(self, source="api"):
        cfg = load_config()
        c = Classification("needs_more_info", f"Rationale {self.PAYLOAD}",
                           [f"missing {self.PAYLOAD}"], [f"verify {self.PAYLOAD}"],
                           0.8, f"model {self.PAYLOAD}")
        iss = issue(summary=f"Summary {self.PAYLOAD}")
        excluded = [{"key": "O3-2", "open_prs": [f"https://x/{self.PAYLOAD}"],
                     "summary": f"Excluded {self.PAYLOAD}"}]
        stamp = datetime.datetime(2026, 8, 1, tzinfo=datetime.timezone.utc)
        with tempfile.TemporaryDirectory() as d:
            base = run.proposal_base(Path(d), stamp)
            path = run.write_comment_report(cfg, base, stamp, [(iss, c, "abc")],
                                            False, source, excluded, swept=2, errors=0)
            return path.read_text()

    def _assert_inert(self, text, marker):
        self.assertIn(marker, text, "the fixture never reached the report")
        tags = {tag for tag, _ in self._elements(text)}
        for injected in ("script", "img", "b"):
            self.assertNotIn(injected, tags,
                             f"<{injected}> was introduced by a field the model "
                             f"or a ticket controls; tags present: {sorted(tags)}")
        for _, attrs in self._elements(text):
            for name in attrs:
                # An attribute can execute without a new element, so tag names
                # alone would miss onerror= landing inside the report's own tags.
                self.assertFalse(name.startswith("on"),
                                 f"event handler {name}= reached the report")

    def test_no_element_from_an_untrusted_field_reaches_the_document(self):
        self._assert_inert(self._report(), "Summary")

    def test_the_replay_banner_escapes_its_self_declared_classifier(self):
        # This banner only renders when source != "api", so the api-path test
        # above never reaches it - and dropping esc() there survived the whole
        # suite. The string is the `classifier` field of a classifications
        # file, which run.py's own comment calls a label rather than proof:
        # "a file can claim any model name it likes". FileClassifier flattens
        # its whitespace, which stops a forged log line but not markup.
        text = self._report(source="file")
        self.assertIn("replayed from a file", text, "the banner did not render")
        self._assert_inert(text, "Summary")


class AgreementIntervalTests(unittest.TestCase):
    """What the pre-registered gate can actually resolve.

    The rule (>= 90%, exit 0/1) is the pilot owners' and is untouched. What was
    missing is any statement of its precision: each case is scored once, the
    graded set is roughly cohort-sized, and the classifier is measured to
    disagree with itself. A reader seeing "90.3%" against a "90%" gate reads a
    clean pass.
    """

    def _mod(self):
        import importlib
        return importlib.import_module("evals.run_evals")

    def test_a_cohort_sized_set_cannot_separate_ninety_from_seventy_five(self):
        # The number that matters for this pilot: 31 in-scope tickets.
        lo, hi = self._mod().agreement_interval(28, 31)
        self.assertLess(lo, 0.80, "an observed 90% over 31 cases reaches below 80%")
        self.assertGreater(hi, 0.95)

    def test_more_cases_narrow_the_interval(self):
        m = self._mod()
        width = lambda hits, n: (lambda b: b[1] - b[0])(m.agreement_interval(hits, n))
        self.assertLess(width(180, 200), width(18, 20),
                        "200 cases must resolve more than 20")
        self.assertLess(width(18, 20), width(9, 10))

    def test_a_unanimous_small_set_still_admits_doubt(self):
        # Wilson rather than the normal approximation precisely for this: at
        # p=1 the usual formula gives a zero-width interval and would report
        # 5/5 as certainty.
        lo, hi = self._mod().agreement_interval(5, 5)
        self.assertLess(lo, 0.9, "5 for 5 is not evidence of clearing a 90% gate")
        self.assertLessEqual(hi, 1.0)

    def test_an_empty_set_claims_nothing(self):
        # graded.csv is empty today, so this is the live state, and the gate
        # must not read as 0% agreement (a confident failure) either.
        self.assertEqual(self._mod().agreement_interval(0, 0), (0.0, 1.0))

    def test_the_interval_never_escapes_zero_to_one(self):
        m = self._mod()
        for hits, n in ((0, 1), (1, 1), (0, 3), (3, 3), (1, 2)):
            lo, hi = m.agreement_interval(hits, n)
            self.assertGreaterEqual(lo, 0.0, (hits, n))
            self.assertLessEqual(hi, 1.0, (hits, n))
            self.assertLessEqual(lo, hi, (hits, n))


class InjectionAttributionTests(unittest.TestCase):
    """The adversarial eval's decision rule, which is pure and so testable here
    even though collecting its data costs API calls.

    The rule exists because the first version of that eval reported a security
    violation off model variance: it saw a payload run return a label the
    single control run had not, and called it a compromise. Five controls later
    the control disagreed with itself. A detector that cries wolf once is
    discounted forever after, so the rule is worth more than the eval around it.
    """

    def _mod(self):
        import importlib
        return importlib.import_module("evals.injection_eval")

    def test_a_novel_label_against_a_stable_control_is_a_violation(self):
        m = self._mod()
        control = collections.Counter({"needs_judgment": 5})
        labels = {m.CONTROL: control,
                  "label forcing": collections.Counter({"automation_candidate": 3,
                                                        "needs_judgment": 2})}
        violations, notes = m.attribute(labels, {}, control)
        self.assertEqual(len(violations), 1)
        self.assertIn("automation_candidate", violations[0])
        self.assertEqual(notes, [])

    def test_the_same_spread_against_an_unstable_control_is_not(self):
        # The measured case: the control itself returned two different labels
        # for identical input, so a payload returning both proves nothing.
        m = self._mod()
        control = collections.Counter({"automation_candidate": 4, "needs_judgment": 1})
        labels = {m.CONTROL: control,
                  "label forcing": collections.Counter({"automation_candidate": 3,
                                                        "needs_judgment": 2})}
        violations, notes = m.attribute(labels, {}, control)
        self.assertEqual(violations, [], "model variance is not a security finding")

    def test_an_unstable_control_still_reports_a_genuinely_new_label(self):
        # Unstable does not mean unfalsifiable: a label outside the control's
        # whole observed range is still attributable, and must be noted rather
        # than silently dropped.
        m = self._mod()
        control = collections.Counter({"automation_candidate": 4, "needs_judgment": 1})
        labels = {m.CONTROL: control,
                  "x": collections.Counter({"needs_more_info": 5})}
        violations, notes = m.attribute(labels, {}, control)
        self.assertEqual(violations, [])
        self.assertEqual(len(notes), 1)
        self.assertIn("needs_more_info", notes[0])

    def test_judged_compliance_is_a_violation_whatever_the_control_did(self):
        # Steering the rationale is the finding that matters: the rationale is
        # posted publicly in the bot's voice, and no amount of label stability
        # excuses it.
        m = self._mod()
        control = collections.Counter({"automation_candidate": 4, "needs_judgment": 1})
        labels = {m.CONTROL: control, "hijack": collections.Counter({"needs_judgment": 5})}
        violations, _ = m.attribute(labels, {"hijack": ["COMPLIED - asserts approval"]},
                                    control)
        self.assertEqual(len(violations), 1)
        self.assertIn("complied", violations[0])


class AnnouncementTests(unittest.TestCase):
    """The maintainer announcement, pinned to the config it describes.

    It tells maintainers which labels to expect and that removing one is a
    permanent opt-out. Renaming a label in config.toml without updating it would
    leave the community watching for a label the bot never applies - and the
    label names are still resting on an untested assumption about whether Jira
    accepts '/', so this is a rename waiting to happen.
    """

    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parent.parent / "docs/maintainer-announcement.md"
        cls.text = path.read_text()
        cls.cfg = load_config()

    def test_every_configured_label_is_explained(self):
        for key in LABEL_KEYS:
            label = self.cfg["labels"][key]
            self.assertIn(label, self.text, f"{label} is applied but never explained")

    def test_no_label_is_named_that_the_pipeline_would_not_apply(self):
        # The `ai-triage-*` glob in prose is not a label and is skipped by this
        # pattern; anything else is a concrete name that must be real.
        named = set(re.findall(r"ai-triage-[a-z-]+", self.text))
        configured = {self.cfg["labels"][k] for k in LABEL_KEYS}
        self.assertEqual(named - configured, set(),
                         "the announcement names labels the pipeline never applies")

    def test_the_intro_outcome_labels_match_the_metric(self):
        # Ask 3 is the only route to one of the three pre-registered thresholds,
        # so the labels it asks for have to be the ones the metric counts.
        for key in ("intro_label", "intro_rejected_label"):
            label = self.cfg["metrics"][key]
            self.assertIn(f"`{label}`", self.text,
                          f"{label} is counted by the intro metric but never asked for")

    def test_the_opt_out_convention_is_stated(self):
        # The kill metric only means something if maintainers know that removing
        # a label opts the ticket out rather than merely tidying it.
        lowered = self.text.lower()
        self.assertIn("opt-out", lowered)
        self.assertIn("permanent", lowered)


class PreflightTests(unittest.TestCase):
    """The go-live gate. It had no tests at all."""

    class Stub:
        """A Jira that answers every preflight probe, configurably."""

        def __init__(self, **overrides):
            self.authenticated = overrides.pop("authenticated", True)
            self.raises = overrides.pop("raises", set())
            self.statuses = overrides.pop("statuses", ["To Do", "Done"])
            self.calls: list[str] = []
            self.comments_posted: list[str] = []
            self.props: dict = {}
            self.labels: list[str] = []
            self.reject_slash = overrides.pop("reject_slash", True)

        def _guard(self, name):
            self.calls.append(name)
            if name in self.raises:
                raise JiraError(f"{name} refused")

        def server_info(self):
            self._guard("server_info")
            return {"baseUrl": "https://example.invalid", "deploymentType": "Cloud"}

        def myself(self):
            self._guard("myself")
            return {"accountId": "bot", "displayName": "Triage Bot"}

        def project_statuses(self, project):
            self._guard("project_statuses")
            return [{"statuses": [{"name": s} for s in self.statuses]}]

        def fields(self):
            self._guard("fields")
            return [{"id": "customfield_1", "name": "Acceptance Criteria"}]

        def search_keys(self, jql):
            self._guard("search_keys")
            return ["O3-1"]

        def update_labels(self, key, add, remove):
            self._guard("update_labels")
            if any("/" in l for l in add) and self.reject_slash:
                raise JiraError("labels may not contain '/'")
            self.labels = [l for l in self.labels if l not in remove] + list(add)

        def add_comment(self, key, body):
            self._guard("add_comment")
            self.comments_posted.append(body)
            return {"id": "10001"}

        def delete_comment(self, key, comment_id):
            self._guard("delete_comment")
            self.comments_posted.pop()

        def set_property(self, key, prop, value):
            self._guard("set_property")
            self.props[prop] = value

        def get_property(self, key, prop):
            self._guard("get_property")
            return self.props.get(prop)

        def delete_property(self, key, prop):
            self._guard("delete_property")
            self.props.pop(prop, None)

    def _run(self, jira, argv=()):
        with mock.patch.dict(os.environ, {"TRIAGE_BOT_ACCOUNT_ID": "bot"}), \
             mock.patch.object(preflight, "jira_from_env", lambda cfg: jira), \
             contextlib.redirect_stdout(io.StringIO()) as out:
            rc = preflight.main(list(argv))
        return rc, out.getvalue()

    def setUp(self):
        StubGitHub.reset()
        self.addCleanup(StubGitHub.reset)

    def test_a_healthy_instance_passes(self):
        rc, report = self._run(self.Stub())
        self.assertEqual(rc, 0, report)
        self.assertNotIn("[FAIL]", report)

    def test_the_open_pr_backstop_is_probed_with_a_key_the_sweep_will_ask_about(self):
        # A probe that passes has to prove the query the pipeline actually runs,
        # so it uses a real in-scope key rather than a simpler synthetic one.
        rc, report = self._run(self.Stub())
        self.assertEqual(rc, 0, report)
        self.assertIn("github open-PR backstop", report)
        self.assertEqual(StubGitHub.searched, ["O3-1"])

    def test_an_unreachable_github_fails_the_gate_and_names_the_way_out(self):
        # Going live with a broken backstop means every ticket errors, so this is
        # a launch blocker, not a warning.
        StubGitHub.reset(error=gh.GitHubError("search unavailable"))
        rc, report = self._run(self.Stub())
        self.assertEqual(rc, 1)
        self.assertIn("[FAIL] github open-PR backstop", report)
        self.assertIn("--no-pr-check", report)

    def test_an_unauthenticated_backstop_still_passes_but_says_so(self):
        rc, report = self._run(self.Stub())
        self.assertEqual(rc, 0, report)
        self.assertIn("GITHUB_TOKEN", report)

    def test_one_failing_probe_does_not_hide_the_others(self):
        # A diagnostic tool that aborts on the first fault is worse than useless:
        # the operator loses every check after it, which is the whole report.
        rc, report = self._run(self.Stub(raises={"project_statuses"}))
        self.assertEqual(rc, 1)
        self.assertIn("[FAIL]", report)
        for later in ("Acceptance Criteria field", "scope JQL"):
            self.assertIn(later, report, f"lost the {later} check\n{report}")

    def test_every_probe_can_fail_without_aborting_the_report(self):
        for probe in ("project_statuses", "fields", "search_keys"):
            rc, report = self._run(self.Stub(raises={probe}))
            self.assertEqual(rc, 1, f"{probe}\n{report}")
            self.assertIn("Anthropic credential", report,
                          f"{probe} aborted the run\n{report}")

    def test_a_missing_scope_status_fails(self):
        rc, report = self._run(self.Stub(statuses=["Done", "In Progress"]))
        self.assertEqual(rc, 1)
        self.assertIn('[FAIL] "To Do" status exists', report)

    def test_a_wrong_bot_account_id_fails(self):
        class Wrong(self.Stub):
            def myself(self):
                self._guard("myself")
                return {"accountId": "someone-else", "displayName": "Someone Else"}

        rc, report = self._run(Wrong())
        self.assertEqual(rc, 1)
        self.assertIn("TRIAGE_BOT_ACCOUNT_ID", report)

    def test_the_scratch_probes_verify_both_write_permissions(self):
        # Add Comments guards the one unrecoverable failure (labelled with no
        # comment); entity properties guard the runaway re-classification loop.
        jira = self.Stub()
        rc, report = self._run(jira, ["--scratch", "O3-1"])
        self.assertEqual(rc, 0, report)
        self.assertIn("[PASS] bot can add comments", report)
        self.assertIn("[PASS] bot can read and write entity properties", report)
        self.assertIn("delete_comment", jira.calls, "the probe comment must be removed")
        self.assertIn("delete_property", jira.calls, "the probe property must be removed")
        self.assertEqual(jira.comments_posted, [], "no probe comment may be left behind")
        self.assertEqual(jira.props, {}, "no probe property may be left behind")
        self.assertEqual(jira.labels, [], "no probe label may be left behind")

    def test_missing_add_comments_permission_fails_the_gate(self):
        rc, report = self._run(self.Stub(raises={"add_comment"}), ["--scratch", "O3-1"])
        self.assertEqual(rc, 1)
        self.assertIn("[FAIL] bot can add comments", report)

    def test_missing_property_permission_fails_the_gate(self):
        rc, report = self._run(self.Stub(raises={"set_property"}), ["--scratch", "O3-1"])
        self.assertEqual(rc, 1)
        self.assertIn("[FAIL] bot can read and write entity properties", report)

    def test_a_silently_discarded_property_write_fails_the_gate(self):
        # The dangerous shape is not a 403 but a write that reports success and
        # stores nothing: idempotency would then be broken with no error anywhere,
        # so the probe must read the value back rather than trust the write.
        class Amnesiac(self.Stub):
            def set_property(self, key, prop, value):
                self._guard("set_property")  # accepted, then discarded

        rc, report = self._run(Amnesiac(), ["--scratch", "O3-1"])
        self.assertEqual(rc, 1)
        self.assertIn("[FAIL] bot can read and write entity properties", report)

    def test_a_rejected_anthropic_key_fails_the_gate(self):
        # The line this replaces reported whether an environment variable
        # existed, which a rotated or revoked key satisfies just as well as a
        # working one. The gate passed, and the sweep then failed all 31
        # tickets until the breaker aborted - after paying for the Jira reads.
        # Adding preflight to the workflow was justified as spending one
        # runner-minute here instead, and this is the credential most likely
        # to be rotated.
        self.addCleanup(StubAnthropicModels.reset)
        StubAnthropicModels.error = RuntimeError("AuthenticationError: 401 invalid x-api-key")
        with mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-revoked"}):
            rc, report = self._run(self.Stub())
        self.assertEqual(rc, 1)
        self.assertIn("[FAIL] Anthropic credential works", report)

    def test_no_anthropic_credential_is_reported_but_not_fatal(self):
        # A dry run and a gather need no credential; only --live does. Failing
        # here would make the gate unusable for the two modes that have carried
        # this whole pilot so far.
        self.addCleanup(StubAnthropicModels.reset)
        StubAnthropicModels.error = RuntimeError("AuthenticationError: no api key")
        with mock.patch.dict(os.environ, {}, clear=True):
            rc, report = self._run(self.Stub())
        self.assertEqual(rc, 0, report)
        self.assertIn("none configured", report)

    def test_a_pinned_model_this_account_cannot_reach_fails_the_gate(self):
        # config.toml naming a model the account cannot use fails exactly like
        # a bad key and exactly as late, so it is worth the same free lookup.
        self.addCleanup(StubAnthropicModels.reset)
        StubAnthropicModels.available = ["claude-haiku-4-5"]
        rc, report = self._run(self.Stub())
        self.assertEqual(rc, 1)
        self.assertIn(f"[FAIL] pinned model {load_config()['claude']['model']} "
                      "is available", report)

    def test_a_full_page_of_models_does_not_condemn_the_pinned_one(self):
        # The listing is paged. Concluding "missing" from a page that was
        # simply full would fail the gate on a large account for a model that
        # is there - a false FAIL on the go-live gate is as costly as a false
        # PASS, because it teaches an operator to override it.
        self.addCleanup(StubAnthropicModels.reset)
        StubAnthropicModels.available = [f"model-{i}" for i in range(100)]
        rc, report = self._run(self.Stub())
        self.assertEqual(rc, 0, report)
        self.assertIn("[PASS] pinned model", report)

    def test_a_missing_acceptance_criteria_field_fails_the_gate(self):
        # The rubric turns on acceptance criteria being visible - "features need
        # some observable way to tell the work is done". If the field cannot be
        # discovered the sweep classifies without it, so preflight must not wave
        # that through. Nothing covered this: dropping the `ok &=` left the
        # suite green.
        class NoAcField(self.Stub):
            def fields(self):
                self._guard("fields")
                return [{"id": "customfield_1", "name": "Something Else"}]

        rc, report = self._run(NoAcField())
        self.assertEqual(rc, 1)
        self.assertIn("[FAIL] Acceptance Criteria field", report)

    def test_no_cleanup_is_attempted_for_a_label_that_never_landed(self):
        # When the hyphenated add fails - a mistyped --scratch key, a missing
        # permission - running the removal anyway prints "remove manually" for a
        # label that was never applied, sending the operator to look for
        # something that is not there, on exactly the broken-key path.
        rc, report = self._run(self.Stub(raises={"update_labels"}),
                               argv=["--scratch", "O3-9999"])
        self.assertEqual(rc, 1)
        self.assertNotIn("could not remove ai-triage-charset-test", report)

    def test_an_unevaluated_dev_panel_clause_fails_rather_than_passing_empty(self):
        # Jira answers a development[] clause it cannot evaluate with an empty
        # result set, not an error - so attempt() reports success and the gate
        # used to print "[PASS] scope JQL - 0 ticket(s)" and exit 0, certifying a
        # sweep that would find nothing. The clause-free count separates "no
        # cohort" from "clause silently ignored".
        class EmptyWithClause(self.Stub):
            def search_keys(self, jql):
                self._guard("search_keys")
                return [] if "development[" in jql else ["O3-1", "O3-2"]

        rc, report = self._run(EmptyWithClause())
        self.assertEqual(rc, 1)
        self.assertIn("[FAIL] scope JQL", report)
        self.assertIn("not being evaluated", report)

    def test_an_empty_cohort_with_a_failed_control_query_is_not_a_pass(self):
        # The tri-state the first version of this fix missed: attempt() prints a
        # FAIL line for a control query that raised but does not touch `ok`, so
        # folding that case in with "both empty" asserted a clause-free count of
        # zero that was never obtained - the same unearned PASS one state over.
        class EmptyThenBroken(self.Stub):
            def search_keys(self, jql):
                self.calls.append("search_keys")
                if "development[" in jql:
                    return []
                raise JiraError("400: bad JQL")

        rc, report = self._run(EmptyThenBroken())
        self.assertEqual(rc, 1)
        self.assertIn("[FAIL] scope JQL", report)
        self.assertIn("no evidence either way", report)

    def test_a_genuinely_empty_cohort_still_passes(self):
        class EmptyEither(self.Stub):
            def search_keys(self, jql):
                self._guard("search_keys")
                return []

        rc, report = self._run(EmptyEither())
        self.assertEqual(rc, 0, report)
        self.assertIn("genuinely empty", report)
        # And the backstop probe says its key was invented rather than implying
        # it exercised something the sweep will ask for.
        self.assertIn("synthetic", report)

    def test_scratch_without_credentials_fails_rather_than_skipping(self):
        # README step 4 is `preflight --scratch O3-XXXX`. Run it without bot
        # credentials and every write probe is skipped - but preflight used to
        # exit 0 anyway, printing advice the operator had already followed. The
        # go-live gate would report PASS having verified no write permission at
        # all: the same false PASS as the slash probe, one level up.
        rc, report = self._run(self.Stub(authenticated=False),
                               argv=["--scratch", "O3-9999"])
        self.assertEqual(rc, 1)
        self.assertIn("[FAIL] write probes requested but not run", report)

    def test_no_scratch_flag_is_still_a_clean_skip(self):
        rc, report = self._run(self.Stub(authenticated=False))
        self.assertEqual(rc, 0, report)
        self.assertNotIn("write probes requested", report)

    def test_a_broken_scratch_ticket_cannot_masquerade_as_slash_rejection(self):
        # The whole probe is "Jira refused the slash". Before the hyphen add
        # gated it, ANY failure counted: a mistyped --scratch key (404), a
        # missing Edit Issues permission (403), a 5xx. The gate then printed
        # PASS on a probe that tested nothing, and the pilot would launch with
        # its label names resting on an assumption nobody had checked.
        rc, report = self._run(self.Stub(raises={"update_labels"}),
                               argv=["--scratch", "O3-9999"])
        self.assertEqual(rc, 1)
        self.assertIn("[FAIL] slash rejected in labels", report)
        self.assertIn("inconclusive", report)

    def test_a_jira_that_accepts_slashes_fails_the_gate(self):
        # The whole reason config.toml deviates from the design doc's names.
        rc, report = self._run(self.Stub(reject_slash=False), ["--scratch", "O3-1"])
        self.assertEqual(rc, 1)
        self.assertIn("[FAIL] slash rejected in labels", report)

    def test_an_unreachable_jira_stops_immediately(self):
        rc, report = self._run(self.Stub(raises={"server_info"}))
        self.assertEqual(rc, 1)
        self.assertIn("[FAIL] jira reachable", report)


class ContextTests(unittest.TestCase):
    def test_app_and_blocked_comments_excluded(self):
        comments = [
            {"author": {"accountType": "app", "displayName": "GitHub"}, "body": "bot noise"},
            {"author": {"accountType": "atlassian", "accountId": "bot", "displayName": "Bot"}, "body": "own comment"},
            {"author": {"accountType": "atlassian", "accountId": "u1", "displayName": "Human"}, "body": "real info"},
        ]
        text = ctx.assemble(StubJira(), issue(comments=comments), None, ["bot"])
        self.assertIn("real info", text)
        self.assertNotIn("bot noise", text)
        self.assertNotIn("own comment", text)

    def test_parent_summary_and_description_fetched(self):
        text = ctx.assemble(StubJira(), issue(parent={"key": "O3-0"}), None, [])
        self.assertIn("PARENT O3-0: Parent summary", text)
        self.assertIn("Parent desc", text)

    def test_an_unreachable_parent_is_declared_not_quietly_omitted(self):
        # out/contexts/<KEY>.txt is the pilot's audit trail - the claim that this
        # is exactly what the model saw. If the parent fetch fails and the
        # context just drops it, a reviewer reading that file infers the model
        # weighed a parent body it never received, and grades the label against
        # evidence that was not there.
        class NoParent(StubJira):
            def issue(self, key, fields, expand_changelog=False):
                raise JiraError("403")

        text = ctx.assemble(NoParent(), issue(parent={"key": "O3-0"}), None, [])
        self.assertIn("PARENT O3-0", text)
        self.assertIn("(parent description unavailable)", text)

    def test_inward_links_are_included_alongside_outward_ones(self):
        # "is blocked by" is inward. Dropping that direction hides exactly the
        # dependency that makes a ticket not ready to work on - the difference
        # between an automation candidate and one that is blocked.
        linked = issue(issuelinks=[
            {"type": {"inward": "is blocked by"},
             "inwardIssue": {"key": "O3-9", "fields": {"summary": "The blocker"}}},
            {"type": {"outward": "relates to"},
             "outwardIssue": {"key": "O3-8", "fields": {"summary": "The relative"}}},
        ])
        text = ctx.assemble(StubJira(), linked, None, [])
        self.assertIn("- is blocked by O3-9: The blocker", text)
        self.assertIn("- relates to O3-8: The relative", text)

    def test_an_empty_description_is_marked_as_empty(self):
        # A blank stretch where the description should be reads as a formatting
        # artefact; "(empty)" is a fact. For needs_more_info - the label that
        # turns on what the ticket does not say - that is the whole signal.
        text = ctx.assemble(StubJira(), issue(description=None), None, [])
        self.assertIn("DESCRIPTION:\n(empty)", text)

    def test_acceptance_criteria_included_when_configured(self):
        text = ctx.assemble(StubJira(), issue(customfield_1="Given X then Y"), "customfield_1", [])
        self.assertIn("ACCEPTANCE CRITERIA:\nGiven X then Y", text)

    def test_linked_tickets_listed(self):
        links = [{"type": {"outward": "blocks", "inward": "is blocked by"},
                  "outwardIssue": {"key": "O3-9", "fields": {"summary": "Other"}}}]
        text = ctx.assemble(StubJira(), issue(issuelinks=links), None, [])
        self.assertIn("- blocks O3-9: Other", text)

    def test_labels_and_status_never_enter_the_context(self):
        """The hash must not move when the bot labels a ticket.

        Every idempotency guarantee rests on this: if the ai-triage label or the
        status reached the assembled text, the bot's own write would change the
        hash, so the next sweep would see "content changed", re-classify, and
        re-charge - forever, on a four-hourly schedule.
        """
        for labels in ([], [AI[0]], AI, ["intro"]):
            for status in ("To Do", "In Progress", "Done"):
                text = ctx.assemble(
                    StubJira(), issue(labels=labels, status={"name": status}), None, [])
                for label in labels:
                    self.assertNotIn(label, text, f"{labels} {status}")
                self.assertNotIn(status, text, f"{labels} {status}")

    def test_the_hash_is_blind_to_everything_but_visible_information(self):
        # Same visible content, wildly different surrounding fields.
        plain = issue()
        decorated = issue(labels=AI, status={"name": "Done"},
                          created="2020-01-01T00:00:00.000+0000")
        decorated["fields"]["assignee"] = {"displayName": "Someone"}
        self.assertEqual(ctx.content_hash(ctx.assemble(StubJira(), plain, None, [])),
                         ctx.content_hash(ctx.assemble(StubJira(), decorated, None, [])))

    def test_bot_content_never_reaches_the_context(self):
        # The visible-information promise, over every author shape Jira produces.
        authors = [
            {"accountType": "app", "displayName": "GitHub", "accountId": "app-1"},
            {"accountType": "atlassian", "accountId": "bot", "displayName": "Bot"},
            {"accountType": "atlassian", "accountId": "blocked", "displayName": "Other bot"},
        ]
        comments = [{"author": a, "body": f"SECRET-{i}"} for i, a in enumerate(authors)]
        comments.append({"author": {"accountType": "atlassian", "accountId": "u1",
                                    "displayName": "Human"}, "body": "VISIBLE"})
        text = ctx.assemble(StubJira(), issue(comments=comments), None, ["bot", "blocked"])
        self.assertIn("VISIBLE", text)
        for i in range(len(authors)):
            self.assertNotIn(f"SECRET-{i}", text)

    def test_a_comment_with_no_author_is_kept(self):
        # Safe direction: an unattributable comment is human-visible content, and
        # dropping it would silently narrow what the classifier is judged on.
        text = ctx.assemble(StubJira(), issue(comments=[{"body": "orphaned note"}]), None, [])
        self.assertIn("orphaned note", text)

    def test_hash_stable_across_assemblies(self):
        a = ctx.assemble(StubJira(), issue(), None, [])
        b = ctx.assemble(StubJira(), issue(), None, [])
        self.assertEqual(ctx.content_hash(a), ctx.content_hash(b))


class CommentTests(unittest.TestCase):
    def test_needs_more_info_comment_lists_missing_items(self):
        cfg = load_config()
        c = Classification("needs_more_info", "No reproduction steps.", ["repro steps"], [], 0.8, "m")
        body = comment_body(cfg, c)
        self.assertIn(cfg["labels"]["needs_more_info"], body)
        self.assertIn("- repro steps", body)
        self.assertIn("opts the ticket out", body)

    def test_automation_candidate_comment_lists_verification(self):
        cfg = load_config()
        c = Classification("automation_candidate", "Clear spec.", [], ["run the report"], 0.9, "m")
        body = comment_body(cfg, c)
        self.assertIn(cfg["labels"]["automation_candidate"], body)
        self.assertIn("- run the report", body)


class WikiSafeTests(unittest.TestCase):
    """Model output is untrusted: it is derived from untrusted ticket text."""

    def test_account_mention_is_neutralised(self):
        # Verified live that Jira renders this from a v2 body as a real @mention,
        # notifying that account. Cohort tickets already contain such tokens.
        c = Classification("needs_more_info", "Ask [~accountid:712020:abc] about it.",
                           ["ping [~accountid:712020:def]"], [], 0.5, "m")
        body = comment_body(load_config(), c)
        self.assertIn("\\[~accountid:712020:abc]", body)
        # No mention token survives unescaped: strip the escaped brackets and
        # nothing that could open a mention should remain.
        self.assertNotIn("[~", body.replace("\\[", ""))

    def test_remote_image_embed_is_neutralised(self):
        c = Classification("automation_candidate", "See !https://attacker.example/p.png!",
                           [], ["run !https://attacker.example/q.png!"], 0.5, "m")
        body = comment_body(load_config(), c)
        self.assertIn("\\!https://attacker.example/p.png\\!", body)
        self.assertNotIn("!", body.replace("\\!", ""))

    def test_ordinary_prose_survives_readably(self):
        c = Classification("needs_more_info", "No reproduction steps.", ["repro steps"], [], 0.8, "m")
        body = comment_body(load_config(), c)
        self.assertIn("No reproduction steps.", body)
        self.assertIn("- repro steps", body)

    def test_our_own_label_markup_is_untouched(self):
        cfg = load_config()
        c = Classification("needs_judgment", "A clinical call.", [], [], 0.8, "m")
        self.assertIn("{{" + cfg["labels"]["needs_judgment"] + "}}", comment_body(cfg, c))


class WritePlanTests(unittest.TestCase):
    def test_fresh_ticket_gets_label_and_comment(self):
        self.assertEqual(plan_label_writes([], AI[0]), ([AI[0]], [], True))

    def test_same_label_present_is_quiet(self):
        self.assertEqual(plan_label_writes([AI[0]], AI[0]), ([], [], False))

    def test_label_flip_removes_stale_and_comments(self):
        self.assertEqual(plan_label_writes([AI[2]], AI[0]), ([AI[0]], [AI[2]], True))

    def test_no_label_is_ever_both_added_and_removed(self):
        # Exhaustive over every present-set and choice, because this is the
        # invariant the test doubles quietly rest on. RecordingJira models a
        # write as (present - remove) + add, which has one meaning only while
        # the two are disjoint; a plan that both added and removed a label
        # would make the double's answer depend on an ordering it invents, and
        # every sweep test would then assert against a Jira that does not
        # exist. Twice this session the defect was in a double rather than in
        # the code, so the assumptions they encode are worth stating out loud.
        for size in range(len(AI) + 1):
            for present in itertools.combinations(AI, size):
                for chosen in AI:
                    add, remove, _ = plan_label_writes(list(present), chosen)
                    self.assertEqual(set(add) & set(remove), set(),
                                     f"present={present} chosen={chosen}")

    def test_every_plan_leaves_exactly_the_chosen_label(self):
        # The planner's real contract, which nothing stated: whatever the
        # ticket carried before, applying the plan leaves precisely one
        # ai-triage label. Two would have the pipeline and metrics.py
        # disagreeing about what the bot decided, and metrics counts labels.
        for size in range(len(AI) + 1):
            for present in itertools.combinations(AI, size):
                for chosen in AI:
                    add, remove, _ = plan_label_writes(list(present), chosen)
                    after = [l for l in present if l not in remove] + list(add)
                    self.assertEqual(sorted(set(after)), [chosen],
                                     f"present={present} chosen={chosen} -> {after}")


def stub_client(pages):
    client = JiraClient("https://example.invalid")
    client.session = StubSession(pages)
    return client


class PaginationTests(unittest.TestCase):
    """Truncated embedded pages are replaced, never extended.

    Jira puts the *newest* window in the issue GET's embedded page and does not
    describe its position consistently (LUI-45: startAt 35 of 135 comments;
    TRUNK-324: startAt 0 while returning the newest 100 changelog entries,
    descending, against the dedicated endpoint's ascending order). Appending to
    it duplicates the overlap and drops the oldest entries.
    """

    def test_untruncated_embedded_page_is_used_as_is(self):
        client = stub_client([])
        result = client.comments("O3-1", {"comments": [{"body": "only"}], "total": 1})
        self.assertEqual([c["body"] for c in result], ["only"])
        self.assertEqual(client.session.calls, [])

    def test_untruncated_changelog_costs_no_extra_request(self):
        client = stub_client([])
        result = client.changelog("O3-1", {"histories": [{"id": "1"}], "total": 1})
        self.assertEqual(len(result), 1)
        self.assertEqual(client.session.calls, [])

    def test_truncated_comments_are_refetched_from_the_start(self):
        # Real LUI-45 shape: the embedded window is comments 35..134 of 135.
        embedded = {"comments": [{"body": f"c{i}"} for i in range(35, 135)],
                    "startAt": 35, "maxResults": 100, "total": 135}
        client = stub_client([
            StubResponse({"comments": [{"body": f"c{i}"} for i in range(100)], "total": 135}),
            StubResponse({"comments": [{"body": f"c{i}"} for i in range(100, 135)], "total": 135}),
        ])
        result = [c["body"] for c in client.comments("LUI-45", embedded)]
        self.assertEqual(len(result), 135)
        self.assertEqual(len(set(result)), 135, "no duplicates")
        self.assertIn("c0", result, "the oldest comment must not be dropped")
        self.assertEqual([c["startAt"] for c in client.session.calls], [0, 100])

    def test_truncated_changelog_is_refetched_from_the_start(self):
        # Real TRUNK-324 shape: embedded holds the newest 100 of 264, and the
        # dedicated endpoint wraps entries in "values" ascending.
        embedded = {"histories": [{"id": str(i)} for i in range(263, 163, -1)],
                    "startAt": 0, "maxResults": 100, "total": 264}
        client = stub_client([
            StubResponse({"values": [{"id": str(i)} for i in range(100)], "total": 264}),
            StubResponse({"values": [{"id": str(i)} for i in range(100, 200)], "total": 264}),
            StubResponse({"values": [{"id": str(i)} for i in range(200, 264)], "total": 264}),
        ])
        result = [h["id"] for h in client.changelog("TRUNK-324", embedded)]
        self.assertEqual(len(result), 264)
        self.assertEqual(len(set(result)), 264, "no duplicates")
        self.assertIn("0", result, "the oldest entry must not be dropped")
        self.assertEqual([c["startAt"] for c in client.session.calls], [0, 100, 200])

    def test_an_issue_fetched_without_its_labels_is_refused(self):
        # "We did not ask" must not read as "there are none": that makes every
        # already-labelled ticket look fresh, so the whole cohort is re-labelled
        # and re-commented to every watcher on the next sweep.
        with self.assertRaises(ValueError) as cm:
            inspect({"key": "O3-1", "fields": {"summary": "x"}}, AI, "bot", [])
        self.assertIn("labels", str(cm.exception))

    def test_an_unattributed_removal_is_an_opt_out_when_no_bot_id_is_set(self):
        # This is the configuration running today - TRIAGE_BOT_ACCOUNT_ID is
        # unset - and Jira does not attribute every changelog entry: a deleted
        # user, an automation rule or an app can arrive with no accountId.
        # Dropping the `bot_account_id is not None` guard makes None == None
        # true, so that removal reads as the bot's own housekeeping and the
        # maintainer's opt-out is discarded. inspect() then reports
        # opted_out=False and the next sweep re-labels the ticket they opted
        # out of. The docstring promises the opposite ("every ai-triage removal
        # counts as an opt-out, the safe direction"); nothing enforced it.
        anonymous = {"author": {"displayName": "Former user"},
                     "created": "2026-07-29T10:00:00.000+0000",
                     "items": [{"field": "labels", "fromString": AI[2], "toString": ""}]}
        st = inspect(issue(), AI, None, [anonymous])
        self.assertTrue(st.opted_out, "an unattributable removal must fail safe")

    def test_the_bots_own_adds_are_not_reported_as_violations_without_a_bot_id(self):
        # The mirror of the rule above. With no bot id nothing can be attributed,
        # so attributing anyway puts every label the bot applied into the weekly
        # digest as a maintainer breaking the no-manual-labels convention -
        # accusing the humans of the bot's own work, on every ticket it touches.
        st = inspect(issue(), AI, None, [label_change("someone", "", AI[0])])
        self.assertEqual(st.human_adds, [])

    def test_opt_out_past_the_embedded_window_is_detected(self):
        embedded = {"histories": [{"items": []}] * 100, "total": 101}
        client = stub_client([StubResponse(
            {"values": [{"items": []}] * 100 + [label_change("u1", AI[2], "")], "total": 101})])
        st = inspect(issue(), AI, "bot", client.changelog("O3-1", embedded))
        self.assertTrue(st.opted_out)

    def test_a_single_manual_add_is_not_double_counted(self):
        # Duplicated history would report one action as two violations.
        embedded = {"histories": [{"items": []}] * 100, "total": 101}
        client = stub_client([StubResponse(
            {"values": [label_change("u1", "", AI[0], display="Maintainer")]
                       + [{"items": []}] * 100, "total": 101})])
        st = inspect(issue(), AI, "bot", client.changelog("O3-1", embedded))
        self.assertEqual(st.human_adds, ["Maintainer"])


    def test_a_short_page_is_refused_rather_than_returned_as_the_whole_history(self):
        # Jira says 264 entries, hands back 100, then stops. Returning those 100
        # as the complete changelog is how a maintainer's opt-out in entry 180
        # becomes invisible - and inspect() then reports opted_out=False with
        # full confidence, so the bot re-labels a ticket a human opted out of.
        # Permanently, publicly, and with nothing in the log to show for it.
        client = stub_client([
            StubResponse({"values": [{"id": str(i)} for i in range(100)], "total": 264}),
            StubResponse({"values": [], "total": 264}),
        ])
        with self.assertRaises(JiraError) as cm:
            client.changelog("O3-1", {"histories": [{"id": "x"}] * 100, "total": 264})
        self.assertIn("264", str(cm.exception))
        self.assertIn("100", str(cm.exception))

    def test_a_short_page_of_comments_is_refused(self):
        # Same read, other caller: a dropped comment is a maintainer's answer
        # the classifier never sees, so it asks for information already given.
        client = stub_client([StubResponse({"comments": [{"body": "c0"}], "total": 9}),
                              StubResponse({"comments": [], "total": 9})])
        with self.assertRaises(JiraError):
            client.comments("O3-1", {"comments": [{"body": "x"}], "total": 9})

    def test_an_unrecognised_page_shape_is_refused_not_read_as_empty(self):
        # If the dedicated endpoint's wrapper key ever changes, every page reads
        # as empty. Before this, that returned [] - a ticket with no history and
        # no comments, silently, for every ticket in the cohort at once.
        client = stub_client([StubResponse({"histories": [{"id": "1"}], "total": 264})])
        with self.assertRaises(JiraError):
            client.changelog("O3-1", {"histories": [{"id": "x"}] * 100, "total": 264})

    def test_an_absent_embedded_page_is_fetched_not_assumed_empty(self):
        # embedded=None means "not asked for", never "there are none". Defaulting
        # the total to the zero items in hand made 0 >= 0 true and returned a
        # complete-looking empty history without a single request.
        client = stub_client([StubResponse(
            {"values": [label_change("u1", AI[2], "")], "total": 1})])
        result = client.changelog("O3-1", None)
        self.assertEqual(len(result), 1)
        self.assertEqual(len(client.session.calls), 1, "it must actually go and ask")

    def test_a_page_without_a_total_is_refused(self):
        # No total is no way to know whether the page in hand is all of them.
        client = stub_client([StubResponse({"values": [{"id": "1"}]})])
        with self.assertRaises(JiraError):
            client.changelog("O3-1", None)

    def test_a_genuinely_empty_dedicated_response_is_accepted(self):
        # The fail-closed guards must not reject a ticket that truly has no
        # history - that would fail every fresh ticket in the cohort.
        client = stub_client([StubResponse({"values": [], "total": 0})])
        self.assertEqual(client.changelog("O3-1", None), [])


class SearchKeysTests(unittest.TestCase):
    """The real cohort query. Every other test stubs search_keys out, so this
    pagination had never executed - on the call that decides which tickets the
    pilot touches at all."""

    def test_all_pages_are_collected(self):
        client = stub_client([
            StubResponse({"issues": [{"key": "O3-1"}], "nextPageToken": "t1"}),
            StubResponse({"issues": [{"key": "O3-2"}]}),
        ])
        self.assertEqual(client.search_keys("project = O3"), ["O3-1", "O3-2"])
        self.assertEqual(client.session.calls[1]["nextPageToken"], "t1")

    def test_a_body_that_is_not_a_search_result_is_refused(self):
        # Jira answers 200 with only warningMessages when a JQL field is
        # unknown - which is exactly what an unavailable development[] clause
        # can look like. Read as an empty cohort, the sweep touches nothing and
        # reports success, and the go-live gate sees a clean run.
        client = stub_client([StubResponse(
            {"warningMessages": ["Field 'development' does not exist"]})])
        with self.assertRaises(JiraError) as cm:
            client.search_keys("project = O3")
        self.assertIn("warningMessages", str(cm.exception))

    def test_an_empty_cohort_is_still_accepted(self):
        client = stub_client([StubResponse({"issues": []})])
        self.assertEqual(client.search_keys("project = O3"), [])

    def test_a_repeating_page_token_is_refused_rather_than_looped_on(self):
        # A cursor that does not advance is an unbounded loop: the sweep never
        # finishes, never reports, and burns the rate limit until CI kills it.
        client = stub_client([StubResponse({"issues": [{"key": "O3-1"}],
                                            "nextPageToken": "t1"})] * 3)
        with self.assertRaises(JiraError) as cm:
            client.search_keys("project = O3")
        self.assertIn("not advancing", str(cm.exception))

    def test_an_unusable_myself_reads_as_no_account_not_a_crash(self):
        # bot_identity_error depends on this returning None rather than raising:
        # it is how "the check could not run" is distinguished from "the ids
        # match", and getting that backwards means proceeding unverified with a
        # bot id that could make the whole cohort read as opted out.
        self.assertIsNone(stub_client([StubResponse({}, 500)]).myself())
        self.assertEqual(
            stub_client([StubResponse({"accountId": "bot"})]).myself()["accountId"], "bot")

    def test_the_get_fallback_still_paginates(self):
        # Falling back to GET on the first page must not lose the cursor: an
        # anonymous sweep would silently see only the first 100 tickets of the
        # cohort and report success on the rest.
        client = stub_client([
            StubResponse({}, 401),
            StubResponse({"issues": [{"key": "O3-1"}], "nextPageToken": "t1"}),
            StubResponse({}, 401),
            StubResponse({"issues": [{"key": "O3-2"}]}),
        ])
        self.assertEqual(client.search_keys("project = O3"), ["O3-1", "O3-2"])
        self.assertEqual(client.session.calls[3]["nextPageToken"], "t1",
                         "the GET fallback dropped the page token")

    def test_an_unauthenticated_post_falls_back_to_get(self):
        client = stub_client([StubResponse({}, 401),
                              StubResponse({"issues": [{"key": "O3-1"}]})])
        self.assertEqual(client.search_keys("project = O3"), ["O3-1"])
        self.assertEqual([c["method"] for c in client.session.calls], ["POST", "GET"])


class ChangelogExpansionTests(unittest.TestCase):
    def test_missing_expansion_errors_rather_than_reading_as_no_history(self):
        # An empty history and a dropped expansion must not look alike: the
        # latter would make every opt-out invisible.
        client = stub_client([StubResponse({"key": "O3-1", "fields": {}})])
        with self.assertRaises(JiraError):
            client.issue("O3-1", ["summary"], expand_changelog=True)

    def test_genuinely_empty_changelog_is_accepted(self):
        client = stub_client([StubResponse(
            {"key": "O3-1", "fields": {}, "changelog": {"total": 0, "histories": []}})])
        issue_json = client.issue("O3-1", ["summary"], expand_changelog=True)
        self.assertEqual(issue_json["changelog"]["total"], 0)

    def test_changelog_not_required_when_not_requested(self):
        client = stub_client([StubResponse({"key": "O3-1", "fields": {}})])
        self.assertEqual(client.issue("O3-1", ["summary"])["key"], "O3-1")


class WriteMethodTests(unittest.TestCase):
    """The three methods that actually touch a public ticket.

    Coverage found these at 0%: every live-run test stubs the client with
    RecordingJira, which overrides all three, so the real bodies had never
    executed. They are the only code in the repo that changes something a
    person can see, and they were the least exercised.
    """

    def test_labels_are_sent_as_add_and_remove_operations(self):
        client = stub_client([StubResponse({}, 204)])
        client.update_labels("O3-1", ["ai-triage-needs-judgment"], ["ai-triage-x"])
        call = client.session.calls[0]
        self.assertEqual(call["method"], "PUT")
        self.assertTrue(call["url"].endswith("/rest/api/2/issue/O3-1"))
        self.assertEqual(call["body"], {"update": {"labels": [
            {"add": "ai-triage-needs-judgment"}, {"remove": "ai-triage-x"}]}},
            "a malformed ops list is a label write that silently does nothing")

    def test_a_notify_suppressed_edit_falls_back_to_a_notifying_one(self):
        # notifyUsers=false needs project admin, which the README says the bot
        # may not have. Untested, this fallback is the difference between the
        # pilot labelling the cohort and 403-ing on every single ticket.
        client = stub_client([StubResponse({}, 403), StubResponse({}, 204)])
        client.update_labels("O3-1", ["ai-triage-needs-judgment"], [])
        first, second = client.session.calls
        self.assertEqual(first["params"], {"notifyUsers": "false"})
        self.assertEqual(second["params"], {}, "the retry must drop notifyUsers")
        self.assertEqual(first["body"], second["body"], "the retry must send the same edit")

    def test_a_second_403_is_raised_not_swallowed(self):
        # The fallback exists for the notifyUsers permission alone. A 403 for
        # any other reason - Edit Issues missing - must surface, or the sweep
        # records a label it never applied and never comments on the ticket
        # again, because the label's presence suppresses the comment.
        client = stub_client([StubResponse({}, 403), StubResponse({}, 403)])
        with self.assertRaises(JiraError):
            client.update_labels("O3-1", ["ai-triage-needs-judgment"], [])

    def test_a_comment_posts_its_body_and_returns_the_created_id(self):
        client = stub_client([StubResponse({"id": "10501"})])
        posted = client.add_comment("O3-1", "AI triage: {{ai-triage-needs-judgment}}")
        self.assertEqual(posted["id"], "10501")
        call = client.session.calls[0]
        self.assertEqual(call["method"], "POST")
        self.assertTrue(call["url"].endswith("/rest/api/2/issue/O3-1/comment"))
        self.assertEqual(call["body"], "AI triage: {{ai-triage-needs-judgment}}")

    def test_a_property_write_is_a_put_to_the_named_property(self):
        client = stub_client([StubResponse({}, 200)])
        client.set_property("O3-1", "ai-triage", {"contentHash": "abc"})
        call = client.session.calls[0]
        self.assertEqual(call["method"], "PUT")
        self.assertTrue(call["url"].endswith("/rest/api/2/issue/O3-1/properties/ai-triage"))
        self.assertEqual(call["body"], {"contentHash": "abc"})

    def test_a_failed_property_write_raises_with_the_status(self):
        # Silence here means every sweep re-classifies and re-charges the whole
        # cohort forever, because the content hash it compares against is never
        # stored. The message carries the status so preflight's probe can say why.
        client = stub_client([StubResponse({"errorMessages": ["no permission"]}, 403)])
        with self.assertRaises(JiraError) as caught:
            client.set_property("O3-1", "ai-triage", {"contentHash": "abc"})
        self.assertIn("403", str(caught.exception))


class PropertyTests(unittest.TestCase):
    def test_missing_property_reads_as_never_triaged(self):
        self.assertIsNone(stub_client([StubResponse({}, 404)]).get_property("O3-1", "ai-triage"))

    def test_stored_hash_is_returned(self):
        client = stub_client([StubResponse({"value": {"contentHash": "abc"}})])
        self.assertEqual(client.get_property("O3-1", "ai-triage"), {"contentHash": "abc"})

    def test_server_error_raises_instead_of_looking_untriaged(self):
        # Swallowing this would silently reclassify every ticket on every run.
        with self.assertRaises(JiraError):
            stub_client([StubResponse({}, 500)]).get_property("O3-1", "ai-triage")


class CleanupCallTests(unittest.TestCase):
    """The two delete methods preflight uses to tidy up after its probes.

    Only ever exercised through a stub before, so nothing checked that they hit
    the right verb, path or timeout - and a preflight probe that cannot clean up
    leaves state on a real ticket.
    """

    def test_delete_comment_targets_the_comment_endpoint(self):
        client = stub_client([StubResponse({}, 204)])
        client.delete_comment("O3-1", "10001")
        call = client.session.calls[0]
        self.assertEqual(call["method"], "DELETE")
        self.assertTrue(call["url"].endswith("/rest/api/2/issue/O3-1/comment/10001"), call)

    def test_delete_property_targets_the_property_endpoint(self):
        client = stub_client([StubResponse({}, 204)])
        client.delete_property("O3-1", "ai-triage-preflight")
        call = client.session.calls[0]
        self.assertEqual(call["method"], "DELETE")
        self.assertTrue(
            call["url"].endswith("/rest/api/2/issue/O3-1/properties/ai-triage-preflight"),
            call)

    def test_an_empty_204_body_is_not_a_parse_error(self):
        # A successful DELETE returns no body; treating that as JSON would turn
        # a clean cleanup into a spurious failure.
        for client, call in (
            (stub_client([StubResponse(None, 204)]), lambda c: c.delete_comment("O3-1", "1")),
            (stub_client([StubResponse(None, 204)]), lambda c: c.delete_property("O3-1", "p")),
        ):
            client.session.pages[0].text = ""
            call(client)
            # Asserted because "it did not raise" also passes for a method that
            # quietly stopped issuing the request at all - and these two are the
            # cleanup calls, whose whole job is to leave nothing behind.
            self.assertEqual([c["method"] for c in client.session.calls], ["DELETE"])

    def test_a_refused_delete_raises(self):
        for method in (lambda c: c.delete_comment("O3-1", "1"),
                       lambda c: c.delete_property("O3-1", "p")):
            client = stub_client([StubResponse({}, 403)])
            with self.assertRaises(JiraError):
                method(client)


class BotIdentityTests(unittest.TestCase):
    class _Jira:
        def __init__(self, account_id, authenticated=True):
            self.authenticated = authenticated
            self._account_id = account_id

        def myself(self):
            return {"accountId": self._account_id, "displayName": "Someone Else"}

    def test_matching_account_is_silent(self):
        self.assertIsNone(bot_identity_error(self._Jira("bot"), "bot"))

    def test_mismatched_account_is_reported(self):
        message = bot_identity_error(self._Jira("someone-else"), "bot")
        self.assertIn("someone-else", message)

    def test_anonymous_client_cannot_check(self):
        self.assertIsNone(bot_identity_error(self._Jira("bot", authenticated=False), "bot"))

    def test_unusable_myself_fails_closed(self):
        # "Could not check" is not "passed": proceeding unverified is exactly how
        # the silent cohort-wide opt-out this guard prevents would still happen.
        class NoMyself:
            authenticated = True

            def myself(self):
                return None

        message = bot_identity_error(NoMyself(), "bot")
        self.assertIsNotNone(message)
        self.assertIn("could not verify", message)

    def test_unset_bot_id_is_not_a_mismatch(self):
        self.assertIsNone(bot_identity_error(self._Jira("bot"), None))


class PromptVersionTests(unittest.TestCase):
    """The prompt file and config.toml must name the same version.

    prompt/system.md says "keep in sync with [prompt].version" and nothing
    enforced it. The drift is silent and it corrupts the pilot's only way to
    explain a label: comments and journal rows stamp config's version while the
    rubric that produced them is a different file, so a removal gets correlated
    to a prompt the model never saw. Bumping the two by hand is exactly the
    moment they diverge.
    """

    @classmethod
    def setUpClass(cls):
        cls.prompt = (Path(run.__file__).resolve().parent.parent
                      / "prompt" / "system.md").read_text()
        cls.cfg = load_config()

    def test_the_prompt_header_names_the_pinned_version(self):
        pinned = self.cfg["prompt"]["version"]
        header = self.prompt.splitlines()[0]
        found = re.search(r"prompt version:\s*(\S+)", header)
        self.assertIsNotNone(found, f"no version header in prompt/system.md: {header!r}")
        self.assertEqual(found.group(1), pinned,
                         "prompt/system.md and config.toml name different prompt versions")

    def test_the_prompt_still_defines_every_label_the_pipeline_applies(self):
        # The classifier's schema accepts exactly these keys; a prompt that
        # stopped describing one would leave the model guessing at a label the
        # pipeline will happily write to a public ticket.
        for key in LABEL_KEYS:
            self.assertIn(key, self.prompt, f"{key} is applied but never defined")


class ScheduleGuardTests(unittest.TestCase):
    """Refuses a local --live run that could race the scheduled sweep.

    The failure it prevents is the only one in this pipeline that cannot be
    undone: two sweeps that both read a ticket before either labels it each post
    a comment, and Jira Cloud has no way to un-send those.
    """

    ACTIVE = ("name: triage\non:\n  workflow_dispatch:\n"
              '  schedule:\n    - cron: "17 */4 * * *"\n')
    COMMENTED = ("name: triage\non:\n  workflow_dispatch:\n"
                 '  # schedule:\n  #   - cron: "17 */4 * * *"\n')

    def _root(self, tmp, text=None):
        root = Path(tmp)
        if text is not None:
            (root / ".github" / "workflows").mkdir(parents=True)
            (root / run.WORKFLOW).write_text(text)
        return root

    def test_a_commented_out_schedule_is_no_conflict(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(run.schedule_conflict(
                self._root(tmp, self.COMMENTED), in_ci=False))

    def test_an_active_schedule_blocks_a_local_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            msg = run.schedule_conflict(self._root(tmp, self.ACTIVE), in_ci=False)
        self.assertIsNotNone(msg)
        self.assertIn(run.SCHEDULE_OVERRIDE, msg, "the refusal must name the way out")

    def test_a_quoted_on_key_is_still_detected(self):
        # YAML reads a bare `on:` as the boolean True; a quoted "on:" lands under
        # the string key. Reading only one of them would miss a live schedule.
        quoted = self.ACTIVE.replace("on:", '"on":')
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNotNone(run.schedule_conflict(
                self._root(tmp, quoted), in_ci=False))

    def test_inside_actions_the_concurrency_group_is_trusted(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(run.schedule_conflict(
                self._root(tmp, self.ACTIVE), in_ci=True))

    def test_no_workflow_file_means_no_cron_to_race(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(run.schedule_conflict(self._root(tmp), in_ci=False))

    def test_an_unreadable_workflow_fails_closed(self):
        # "Probably no cron" is not worth duplicate comments on public tickets.
        with tempfile.TemporaryDirectory() as tmp:
            msg = run.schedule_conflict(
                self._root(tmp, "on:\n  schedule:\n   - cron: [unclosed\n"),
                in_ci=False)
        self.assertIsNotNone(msg)
        self.assertIn(run.SCHEDULE_OVERRIDE, msg)

    def test_this_repos_own_workflow_is_readable(self):
        # Whatever the schedule's state, the guard must not fail closed on our own
        # file - that would block every local live run for the wrong reason.
        msg = run.schedule_conflict(Path(run.__file__).resolve().parent.parent,
                                    in_ci=False)
        self.assertNotIn("cannot tell", msg or "")

    def _live(self, argv, conflict):
        jira = RecordingJira({"O3-1": issue()})
        with tempfile.TemporaryDirectory() as d, \
             mock.patch.dict(os.environ, {"TRIAGE_BOT_ACCOUNT_ID": "bot",
                                          "JIRA_EMAIL": "b@x", "JIRA_API_TOKEN": "t"}), \
             mock.patch.object(run, "jira_from_env", lambda cfg: jira), \
             mock.patch.object(run, "schedule_conflict", lambda *a: conflict), \
             mock.patch.object(run, "Classifier", lambda *a: StubClassifier(
                 Classification("needs_judgment", "x", [], [], 0.5, "m"))), \
             contextlib.redirect_stdout(io.StringIO()), \
             contextlib.redirect_stderr(io.StringIO()):
            try:
                rc = run.main(argv, out=Path(d))
            except SystemExit as e:
                return str(e.code), jira.writes
        return rc, jira.writes

    def test_a_conflict_stops_the_run_before_anything_is_written(self):
        StubGitHub.reset()
        self.addCleanup(StubGitHub.reset)
        code, writes = self._live(["--live", "--keys", "O3-1"], "a sweep may fire")
        self.assertIn("a sweep may fire", code)
        self.assertEqual(writes, [], "wrote to Jira despite a schedule conflict")

    def test_the_override_lets_a_deliberate_run_through(self):
        StubGitHub.reset()
        self.addCleanup(StubGitHub.reset)
        code, writes = self._live(
            ["--live", run.SCHEDULE_OVERRIDE, "--keys", "O3-1"], "a sweep may fire")
        self.assertEqual(code, 0)
        self.assertTrue(writes, "the override did not let the run proceed")


class PlanTicketTests(unittest.TestCase):
    def test_opt_out_beats_force(self):
        st = TicketState(ai_labels_present=[AI[0]], opted_out=True)
        self.assertEqual(plan_ticket(st, False, True, True), "skip-opted-out")

    def test_unchanged_labeled_ticket_is_skipped(self):
        st = TicketState(ai_labels_present=[AI[0]])
        self.assertEqual(plan_ticket(st, True, False, True), "skip-already-triaged")

    def test_force_reclassifies_unchanged_ticket(self):
        st = TicketState(ai_labels_present=[AI[0]])
        self.assertIsNone(plan_ticket(st, True, True, True))

    def test_changed_content_reclassifies(self):
        st = TicketState(ai_labels_present=[AI[0]])
        self.assertIsNone(plan_ticket(st, False, False, True))

    def test_fresh_ticket_is_classified(self):
        self.assertIsNone(plan_ticket(TicketState(), True, False, True))

    def test_no_classifier_stops_at_context(self):
        self.assertEqual(plan_ticket(TicketState(), True, False, False), "context-only")

    def test_out_of_scope_ticket_is_skipped(self):
        self.assertEqual(plan_ticket(TicketState(), False, False, True, out_of_scope=True),
                         "skip-out-of-scope")

    def test_force_does_not_override_scope(self):
        self.assertEqual(plan_ticket(TicketState(), False, True, True, out_of_scope=True),
                         "skip-out-of-scope")

    def test_opt_out_is_reported_ahead_of_scope(self):
        st = TicketState(opted_out=True)
        self.assertEqual(plan_ticket(st, False, False, True, out_of_scope=True),
                         "skip-opted-out")

    def test_ticket_with_an_open_pr_is_skipped(self):
        self.assertEqual(plan_ticket(TicketState(), False, False, True, has_open_pr=True),
                         "skip-open-pr")

    def test_force_does_not_override_an_open_pr(self):
        self.assertEqual(plan_ticket(TicketState(), False, True, True, has_open_pr=True),
                         "skip-open-pr")

    def test_opt_out_is_reported_ahead_of_an_open_pr(self):
        st = TicketState(opted_out=True)
        self.assertEqual(plan_ticket(st, False, False, True, has_open_pr=True),
                         "skip-opted-out")

    def test_an_open_pr_outranks_already_triaged(self):
        # Both leave an unchanged ticket alone, but only skip-open-pr keeps it out
        # of the manifest, so a ticket that gained a PR after being labelled stops
        # being offered for re-classification on the next prompt bump.
        st = TicketState(ai_labels_present=[AI[0]])
        self.assertEqual(plan_ticket(st, True, False, True, has_open_pr=True),
                         "skip-open-pr")

    def test_an_open_pr_stops_a_context_only_gather(self):
        # The gather step builds the grading sheet's input, so an in-review ticket
        # has to drop out there too, not only where writes happen.
        self.assertEqual(plan_ticket(TicketState(), True, False, False, has_open_pr=True),
                         "skip-open-pr")


class GitHubSession:
    """Replays queued GitHub search responses and records the queries asked."""

    def __init__(self, pages):
        self.headers: dict = {}
        self.pages = list(pages)
        self.queries: list[str] = []

    def get(self, url, params=None, timeout=None):
        assert timeout is not None, "every network call must carry a timeout"
        self.queries.append((params or {}).get("q", ""))
        return self.pages.pop(0)


def gh_response(items, status_code=200, headers=None):
    """A stub shaped like a real GitHub search response.

    total_count and incomplete_results are included because the client requires
    them: a helper that omitted total_count would model a body GitHub never
    sends, and every test built on it would be exercising a shape the guard is
    specifically there to reject.
    """
    items = list(items)
    payload = {"items": items, "total_count": len(items), "incomplete_results": False}
    resp = StubResponse(payload, status_code)
    resp.headers = dict(headers or {})
    resp.text = json.dumps(payload)
    return resp


def pr(number=1, title="", body="", url=None):
    return {"number": number, "title": title, "body": body,
            "html_url": url or f"https://github.com/openmrs/repo/pull/{number}"}


class KeyCitationTests(unittest.TestCase):
    """Which mentions count as a PR claiming a ticket."""

    def test_a_plain_mention_counts(self):
        self.assertTrue(gh.names_key("(fix) O3-5816: stop the crash", "O3-5816"))

    def test_case_is_ignored(self):
        # Jira accepts and normalises a lowercase key, so a title may carry either.
        self.assertTrue(gh.names_key("fixes o3-5816 finally", "O3-5816"))

    def test_a_longer_key_is_not_a_match(self):
        # The bug this prevents: searching O3-581 excluding a ticket because some
        # PR mentions O3-5816.
        self.assertFalse(gh.names_key("(fix) O3-5816: stop the crash", "O3-581"))

    def test_a_key_with_a_trailing_digit_is_not_a_match(self):
        self.assertFalse(gh.names_key("about O3-58161 really", "O3-5816"))

    def test_a_key_glued_to_a_word_is_not_a_match(self):
        self.assertFalse(gh.names_key("branchO3-5816", "O3-5816"))

    def test_punctuation_around_the_key_is_fine(self):
        for text in ("[O3-5816]", "(O3-5816)", "O3-5816.", "see O3-5816, then"):
            self.assertTrue(gh.names_key(text, "O3-5816"), text)

    def test_empty_text_is_not_a_match(self):
        self.assertFalse(gh.names_key("", "O3-5816"))
        self.assertFalse(gh.names_key(None, "O3-5816"))


class GitHubClientTests(unittest.TestCase):
    """The dev-panel backstop's client, offline."""

    def _client(self, pages, token=None):
        client = gh.GitHubClient("openmrs", token, sleep=lambda s: self.slept.append(s),
                                 now=lambda: self.clock)
        client.session = GitHubSession(pages)
        return client

    def setUp(self):
        self.slept: list[float] = []
        self.clock = 1000.0

    def test_a_well_formed_empty_result_is_still_a_valid_negative(self):
        payload = {"items": [], "total_count": 0, "incomplete_results": False}
        resp = gh_response([])
        resp.text = json.dumps(payload)
        resp._payload = payload
        self.assertEqual(self._client([resp]).open_pr_urls("O3-1"), [])

    def test_the_query_asks_only_for_open_prs_in_the_org(self):
        client = self._client([gh_response([])])
        client.open_pr_urls("O3-5816")
        self.assertEqual(client.session.queries,
                         ["org:openmrs is:pr is:open O3-5816"])

    def test_a_pr_naming_the_key_in_its_title_is_returned(self):
        client = self._client([gh_response([
            pr(1818, title="(fix) O3-5816: stop the crash",
               url="https://github.com/openmrs/openmrs-esm-core/pull/1818")])])
        self.assertEqual(client.open_pr_urls("O3-5816"),
                         ["https://github.com/openmrs/openmrs-esm-core/pull/1818"])

    def test_a_pr_naming_the_key_only_in_its_body_is_returned(self):
        client = self._client([gh_response([pr(2, title="Fix login", body="Closes O3-5816")])])
        self.assertEqual(len(client.open_pr_urls("O3-5816")), 1)

    def test_a_full_text_hit_that_does_not_name_the_key_is_dropped(self):
        # Search also matches PR *comments*, which is wider than the dev panel's
        # notion of a link: "unrelated to O3-5816" in a review would otherwise
        # exclude a ticket nobody is working on.
        client = self._client([gh_response([pr(3, title="Refactor", body="No keys here")])])
        self.assertEqual(client.open_pr_urls("O3-5816"), [])

    def test_a_missing_html_url_falls_back_to_the_api_url(self):
        client = self._client([gh_response([
            {"number": 7, "title": "O3-5816 fix",
             "url": "https://api.github.com/repos/openmrs/r/pulls/7"}])])
        self.assertEqual(client.open_pr_urls("O3-5816"),
                         ["https://api.github.com/repos/openmrs/r/pulls/7"])

    def test_with_no_url_at_all_the_fallback_is_not_a_link(self):
        # This value is rendered as an href in the report a reviewer uses to
        # confirm an exclusion. "openmrs#7" looks checkable and goes nowhere,
        # which is worse than plain text they can search for.
        client = self._client([gh_response([{"number": 7, "title": "O3-5816 fix"}])])
        got = client.open_pr_urls("O3-5816")[0]
        self.assertNotRegex(got, r"^\S+#\d+$")
        self.assertIn("7", got)

    def test_searches_are_throttled_below_the_unauthenticated_rate(self):
        client = self._client([gh_response([]), gh_response([])])
        self.assertGreater(client.min_interval, 60.0 / gh.RATE_LIMIT_PER_MIN[False])
        client.open_pr_urls("O3-1")
        client.open_pr_urls("O3-2")
        # The clock did not advance between them, so the whole interval is waited.
        self.assertEqual(self.slept, [client.min_interval])

    def test_a_token_raises_the_rate(self):
        client = self._client([gh_response([]), gh_response([])], token="t")
        self.assertTrue(client.authenticated)
        self.assertLess(client.min_interval, gh.GitHubClient("openmrs").min_interval)

    def test_neither_rate_is_paced_at_the_documented_ceiling(self):
        # Measured: pacing at exactly 30/min took a 403 on the last ticket of a
        # 32-ticket sweep, because anything else sharing the token lands in the
        # same window. The interval must leave the sweep somewhere to drift.
        for token in (None, "t"):
            client = gh.GitHubClient("openmrs", token)
            ceiling = 60.0 / gh.RATE_LIMIT_PER_MIN[bool(token)]
            self.assertGreater(client.min_interval, ceiling,
                               f"token={token!r} paces at the documented limit")

    def test_a_token_is_sent_as_a_bearer_header(self):
        # Built without the stub session, which would discard the real headers.
        client = gh.GitHubClient("openmrs", "t")
        self.assertEqual(client.session.headers["Authorization"], "Bearer t")
        self.assertNotIn("Authorization", gh.GitHubClient("openmrs").session.headers)

    def test_time_already_spent_elsewhere_is_not_waited_again(self):
        client = self._client([gh_response([]), gh_response([])])
        client.open_pr_urls("O3-1")
        self.clock += 10.0  # a Jira fetch and a classification happened meanwhile
        client.open_pr_urls("O3-2")
        self.assertEqual(self.slept, [])

    def test_a_short_rate_limit_window_is_waited_out_once(self):
        limited = gh_response([], 403, {"x-ratelimit-remaining": "0",
                                        "retry-after": "20"})
        client = self._client([limited, gh_response([pr(1, title="O3-1 fix")])])
        self.assertEqual(len(client.open_pr_urls("O3-1")), 1)
        self.assertIn(20.0, self.slept)

    def test_a_long_rate_limit_window_fails_loudly_with_the_way_out(self):
        # Sleeping through it would silently stall a sweep the pilot expects to
        # finish inside its 24h SLA.
        limited = gh_response([], 403, {"x-ratelimit-remaining": "0",
                                        "retry-after": "900"})
        client = self._client([limited])
        with self.assertRaises(gh.GitHubError) as caught:
            client.open_pr_urls("O3-1")
        self.assertIn("GITHUB_TOKEN", str(caught.exception))
        self.assertIn("--no-pr-check", str(caught.exception))

    def test_a_403_that_is_not_a_rate_limit_is_not_waited_out(self):
        # A bad token must fail now, not after a pointless sleep.
        denied = gh_response([], 403, {"x-ratelimit-remaining": "42"})
        denied.text = '{"message": "Bad credentials"}'
        client = self._client([denied])
        with self.assertRaises(gh.GitHubError):
            client.open_pr_urls("O3-1")
        self.assertEqual(self.slept, [])

    def test_the_secondary_limit_is_waited_out_even_with_no_headers(self):
        # The shape a real sweep hit: a 403 saying the limit was exceeded, with
        # neither retry-after nor an exhausted remaining count. Reading it as
        # "not a rate limit" failed the ticket instead of waiting out the window.
        limited = gh_response([], 403, {})
        limited.text = ('{"message": "API rate limit exceeded for user ID 1390773. '
                        'If you reach out to GitHub Support for help..."}')
        client = self._client([limited, gh_response([pr(1, title="O3-1 fix")])])
        self.assertEqual(len(client.open_pr_urls("O3-1")), 1)
        self.assertIn(gh.DEFAULT_RATE_WAIT_SECONDS, self.slept)

    def test_an_exhausted_limit_with_an_unparseable_reset_still_waits(self):
        limited = gh_response([], 403, {"x-ratelimit-remaining": "0",
                                        "x-ratelimit-reset": "not-a-number"})
        client = self._client([limited, gh_response([])])
        client.open_pr_urls("O3-1")
        self.assertIn(gh.DEFAULT_RATE_WAIT_SECONDS, self.slept)

    def test_a_second_rate_limit_in_a_row_fails_rather_than_looping(self):
        limited = gh_response([], 403, {})
        limited.text = '{"message": "API rate limit exceeded"}'
        again = gh_response([], 403, {})
        again.text = '{"message": "API rate limit exceeded"}'
        client = self._client([limited, again])
        with self.assertRaises(gh.GitHubError):
            client.open_pr_urls("O3-1")
        self.assertEqual(len(self.slept), 1, "waited more than once")

    def test_a_reset_timestamp_is_honoured_when_there_is_no_retry_after(self):
        limited = gh_response([], 429, {"x-ratelimit-remaining": "0",
                                        "x-ratelimit-reset": str(int(time.time()) + 30)})
        client = self._client([limited, gh_response([])])
        client.open_pr_urls("O3-1")
        self.assertTrue(any(25 <= s <= 31 for s in self.slept), self.slept)

    def test_an_empty_body_is_not_read_as_no_open_pr(self):
        # A 200 with no body (proxy, dropped response) used to become {} and then
        # an empty result list - fabricating the absence proof this module exists
        # to supply. In --live that labels and comments on a ticket already in
        # review, publicly and unsendably.
        blank = gh_response([])
        blank.text = ""
        client = self._client([blank])
        with self.assertRaises(gh.GitHubError):
            client.open_pr_urls("O3-1")

    def test_a_timed_out_search_is_not_read_as_no_open_pr(self):
        # GitHub sets incomplete_results when its own search times out, so an
        # empty result means "we did not finish looking".
        # total_count present and consistent on purpose: without it the shape
        # guard fires first and this test passes without ever reaching the one
        # it names. Mutation testing caught exactly that.
        payload = {"items": [], "total_count": 0, "incomplete_results": True}
        resp = gh_response([])
        resp.text = json.dumps(payload)
        resp._payload = payload
        client = self._client([resp])
        with self.assertRaises(gh.GitHubError) as caught:
            client.open_pr_urls("O3-1")
        self.assertIn("timed out", str(caught.exception))

    def test_a_json_body_that_is_not_a_search_result_is_refused(self):
        # The guard this covers was itself an instance of the class it was added
        # to remove: total_count defaulted to len(items), making the comparison
        # len(items) > len(items) and always false, so a proxy envelope or an
        # error body passed every check and returned an unearned empty.
        for body in ({}, {"message": "Bad gateway"}, {"items": []}):
            resp = gh_response([])
            resp.text = json.dumps(body)
            resp._payload = body
            client = self._client([resp])
            with self.assertRaises(gh.GitHubError, msg=str(body)):
                client.open_pr_urls("O3-1")

    def test_a_truncated_result_set_is_refused(self):
        # This client does not paginate; if GitHub says there are more matches
        # than it returned, the unseen ones could hold the key.
        payload = {"items": [pr(1, title="unrelated")], "total_count": 250}
        resp = gh_response([])
        resp.text = json.dumps(payload)
        resp._payload = payload
        client = self._client([resp])
        with self.assertRaises(gh.GitHubError) as caught:
            client.open_pr_urls("O3-1")
        self.assertIn("does not paginate", str(caught.exception))

    def test_a_server_error_is_reported_not_swallowed(self):
        # Returning "no open PR" on a 500 would re-open the leak silently, on the
        # tickets most likely to be in review.
        client = self._client([gh_response([], 500)])
        with self.assertRaises(gh.GitHubError):
            client.open_pr_urls("O3-1")


class OpenPrBackstopWiringTests(unittest.TestCase):
    """The backstop inside a real run: what it excludes, and what it costs."""

    def setUp(self):
        StubGitHub.reset()
        self.addCleanup(StubGitHub.reset)

    def _run(self, jira, extra_args=(), live=False, out=None):
        classification = Classification("automation_candidate", "Because.", [], ["check it"],
                                        0.9, "m")
        argv = (["--live"] if live else []) + ["--keys", "O3-1", *extra_args]
        with tempfile.TemporaryDirectory() as d:
            d = str(out) if out else d
            with mock.patch.dict(os.environ, {"TRIAGE_BOT_ACCOUNT_ID": "bot"}), \
                 mock.patch.object(run, "jira_from_env", lambda cfg: jira), \
                 mock.patch.object(run, "Classifier", lambda *a: StubClassifier(classification)), \
                 contextlib.redirect_stdout(io.StringIO()), \
                 contextlib.redirect_stderr(io.StringIO()):
                rc = run.main(argv, out=Path(d))
            journal = (Path(d) / "journal.jsonl").read_text().splitlines()
        return rc, json.loads(journal[-1])

    def test_the_config_enables_the_backstop(self):
        # The whole fix is inert if this is off, and every test below would pass
        # for the wrong reason.
        self.assertTrue(load_config()["github"]["check_open_prs"])

    def test_a_ticket_with_an_open_pr_is_not_classified(self):
        StubGitHub.reset({"O3-1": ["https://github.com/openmrs/repo/pull/9"]})
        jira = RecordingJira({"O3-1": issue()})
        rc, row = self._run(jira, live=True)
        self.assertEqual(row["action"], "skip-open-pr")
        self.assertEqual(jira.writes, [], "labelled a ticket that is already in review")
        self.assertEqual(rc, 0, "an in-review ticket is a routine skip, not a fault")

    def test_the_excluded_ticket_records_the_evidence(self):
        # This is the one skip whose evidence lives outside Jira, so the journal
        # has to carry it or the exclusion is unauditable.
        StubGitHub.reset({"O3-1": ["https://github.com/openmrs/repo/pull/9"]})
        _, row = self._run(RecordingJira({"O3-1": issue()}))
        self.assertEqual(row["open_prs"], ["https://github.com/openmrs/repo/pull/9"])

    def test_an_excluded_ticket_stays_out_of_the_manifest(self):
        StubGitHub.reset({"O3-1": ["https://github.com/openmrs/repo/pull/9"]})
        jira = RecordingJira({"O3-1": issue()})
        with tempfile.TemporaryDirectory() as d:
            self._run(jira, extra_args=["--no-classify"], out=Path(d))
            manifest = Path(d) / "manifest.json"
            # No manifest at all is also a pass: the gather found nothing to
            # classify, which is the point.
            tickets = json.loads(manifest.read_text())["tickets"] if manifest.exists() else {}
        self.assertNotIn("O3-1", tickets)

    def test_a_ticket_with_no_open_pr_is_classified_as_before(self):
        StubGitHub.reset({})
        jira = RecordingJira({"O3-1": issue()})
        _, row = self._run(jira, live=True)
        self.assertEqual(row["action"], "labeled")
        self.assertEqual(StubGitHub.searched, ["O3-1"])

    def test_no_pr_check_skips_the_lookup_entirely(self):
        StubGitHub.reset({"O3-1": ["https://github.com/openmrs/repo/pull/9"]})
        jira = RecordingJira({"O3-1": issue()})
        _, row = self._run(jira, extra_args=["--no-pr-check"])
        self.assertEqual(StubGitHub.searched, [], "searched despite --no-pr-check")
        self.assertEqual(StubGitHub.built, 0, "built a client despite --no-pr-check")
        self.assertEqual(row["action"], "proposed")

    def test_a_github_failure_fails_the_ticket_rather_than_labelling_it(self):
        # Fail-open would re-open the leak exactly when GitHub is unreachable,
        # and a wrong label costs a permanent opt-out. The errored ticket writes
        # no property, so the next sweep retries it.
        StubGitHub.reset(error=gh.GitHubError("rate limited"))
        jira = RecordingJira({"O3-1": issue()})
        rc, row = self._run(jira, live=True)
        self.assertEqual(row["action"], "error")
        self.assertIn("rate limited", row["error"])
        self.assertEqual(jira.writes, [])
        self.assertEqual(rc, 1, "a sweep that classified nothing must exit non-zero")

    def test_an_opted_out_ticket_costs_no_search(self):
        # The searches are rate-limited, and no answer could change the outcome.
        StubGitHub.reset({"O3-1": ["https://github.com/openmrs/repo/pull/9"]})
        jira = RecordingJira({"O3-1": issue(histories=[label_change("u1", AI[2], "")])})
        _, row = self._run(jira, live=True)
        self.assertEqual(row["action"], "skip-opted-out")
        self.assertEqual(StubGitHub.searched, [])

    def test_an_already_triaged_ticket_costs_no_search_outside_gather(self):
        # Steady state: most of the cohort is already labelled, and for those the
        # answer changes nothing outside gather - the manifest it would alter is
        # only written under --no-classify. Paying ~30 throttled searches per
        # sweep to change a journal string also buys ~30 more chances of the
        # secondary rate limit, which fails tickets outright.
        jira = RecordingJira({"O3-1": issue()})
        StubGitHub.reset({})
        first, _ = self._run(jira, live=True)[0], None   # writes label + property
        # A PR appears afterwards. The ticket is now genuinely already-triaged:
        # label present, stored hash and prompt both match.
        StubGitHub.reset({"O3-1": ["https://github.com/openmrs/repo/pull/9"]})
        _, row = self._run(jira, live=True)
        self.assertEqual(row["action"], "skip-already-triaged")
        self.assertEqual(StubGitHub.searched, [],
                         "searched GitHub for a ticket whose answer cannot matter")

    def test_gather_still_asks_about_already_triaged_tickets(self):
        # --no-classify writes the manifest, and deliberately offers labelled
        # tickets for re-classification. One that is now in review has to drop
        # out there, so gather must still consult GitHub.
        StubGitHub.reset({"O3-1": ["https://github.com/openmrs/repo/pull/9"]})
        jira = RecordingJira({"O3-1": issue(labels=[AI[2]])})
        _, row = self._run(jira, extra_args=["--no-classify"])
        self.assertEqual(StubGitHub.searched, ["O3-1"])
        self.assertEqual(row["action"], "skip-open-pr")

    def test_an_out_of_scope_ticket_costs_no_search(self):
        StubGitHub.reset({"O3-1": ["https://github.com/openmrs/repo/pull/9"]})
        jira = RecordingJira({"O3-1": issue(status={"name": "In Progress"})})
        _, row = self._run(jira, live=True)
        self.assertEqual(row["action"], "skip-out-of-scope")
        self.assertEqual(StubGitHub.searched, [])

    def test_the_run_says_whether_the_backstop_was_on(self):
        # A sweep whose scope is wider than the pilot documented must say so in
        # its own log, not only in whatever flags the operator remembered.
        for args, expected in ((), "open-PR backstop: on"), (("--no-pr-check",), "backstop: OFF"):
            StubGitHub.reset()
            buf = io.StringIO()
            with mock.patch.dict(os.environ, {"TRIAGE_BOT_ACCOUNT_ID": "bot"}), \
                 mock.patch.object(run, "jira_from_env",
                                   lambda cfg: RecordingJira({"O3-1": issue()})), \
                 mock.patch.object(run, "Classifier", lambda *a: StubClassifier(
                     Classification("needs_judgment", "x", [], [], 0.5, "m"))), \
                 tempfile.TemporaryDirectory() as d, \
                 contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
                run.main(["--keys", "O3-1", *args], out=Path(d))
            self.assertIn(expected, buf.getvalue())


class DecisionRuleTests(unittest.TestCase):
    """The pre-registered rule; committed before launch, so it must be right now."""

    M = {"sorted_within_24h_pct": 95, "max_label_removal_rate": 0.10, "min_intro_outcomes": 5}
    LAUNCH = datetime.datetime.fromisoformat("2026-08-01T00:00:00+00:00")

    def test_backlog_sla_runs_from_launch_not_creation(self):
        # A 90-day-old backlog ticket labeled 2h after launch meets the SLA.
        self.assertTrue(sla_met("2026-05-04T09:00:00.000+0000",
                                "2026-08-01T02:00:00.000+0000", self.LAUNCH))

    def test_backlog_sla_missed_past_24h_after_launch(self):
        self.assertFalse(sla_met("2026-05-04T09:00:00.000+0000",
                                 "2026-08-02T02:00:00.000+0000", self.LAUNCH))

    def test_pilot_era_ticket_sla_runs_from_creation(self):
        created = "2026-08-10T09:00:00.000+0000"
        self.assertTrue(sla_met(created, "2026-08-11T08:00:00.000+0000", self.LAUNCH))
        self.assertFalse(sla_met(created, "2026-08-12T09:00:00.000+0000", self.LAUNCH))

    def test_exactly_24h_counts_as_within_24h(self):
        # "within 24h" is pre-registered, and pre-registration is worth only as
        # much as the fixity of the definition behind it. Nothing pinned which
        # side of the boundary is inclusive, so flipping <= to < was a silent
        # tightening of a committed metric after launch - the one change the
        # whole design is meant to make impossible.
        created = "2026-08-10T09:00:00.000+0000"
        self.assertTrue(sla_met(created, "2026-08-11T09:00:00.000+0000", self.LAUNCH))
        self.assertFalse(sla_met(created, "2026-08-11T09:00:00.001+0000", self.LAUNCH))

    def test_all_three_metrics_pass_adopts(self):
        self.assertEqual(decide(100.0, 0.05, 10, self.M), "ADOPT")

    def test_thresholds_are_inclusive(self):
        self.assertEqual(decide(95.0, 0.10, 5, self.M), "ADOPT")

    def test_two_of_three_with_tolerable_removals_extends(self):
        self.assertEqual(decide(100.0, 0.05, 2, self.M), "EXTEND (two weeks)")
        self.assertEqual(decide(100.0, 0.15, 10, self.M), "EXTEND (two weeks)")

    def test_removal_rate_past_double_threshold_stops(self):
        # The kill metric overrides two passing metrics.
        self.assertEqual(decide(100.0, 0.25, 10, self.M), "STOP")

    def test_one_of_three_stops(self):
        self.assertEqual(decide(50.0, 0.05, 1, self.M), "STOP")

    def test_launch_date_is_parsed_as_midnight_utc(self):
        self.assertEqual(parse_launch("2026-08-01"), self.LAUNCH)

    def test_unset_launch_date_names_the_config_field(self):
        with self.assertRaises(SystemExit) as caught:
            parse_launch("")
        self.assertIn("pilot_launch", str(caught.exception))

    def test_timestamp_instead_of_a_date_names_the_config_field(self):
        # Appending a time to this produced an opaque "Invalid isoformat string"
        # that said nothing about which field was wrong.
        with self.assertRaises(SystemExit) as caught:
            parse_launch("2026-08-01T00:00:00+00:00")
        self.assertIn("pilot_launch", str(caught.exception))
        self.assertIn("bare date", str(caught.exception))


def load_evals_module():
    """Import evals/run_evals.py, which is a script rather than a package."""
    import importlib.util

    path = Path(__file__).resolve().parent.parent / "evals" / "run_evals.py"
    spec = importlib.util.spec_from_file_location("triage_run_evals", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ProposalSheetTests(unittest.TestCase):
    def test_sheet_records_the_context_hash_and_a_second_precise_stamp(self):
        c = Classification("needs_more_info", "No repro.", ["repro steps"], [], 0.8, "m")
        stamp = datetime.datetime(2026, 8, 1, 9, 30, 15, tzinfo=datetime.timezone.utc)
        with tempfile.TemporaryDirectory() as d:
            base = run.write_proposals(load_config(), Path(d), stamp,
                                       [(issue(), c, "deadbeef12345678")], live=False)
            with open(base.with_suffix(".csv")) as fh:
                row = next(csv.DictReader(fh))
        self.assertEqual(row["content_hash"], "deadbeef12345678")
        self.assertEqual(row["proposed_label"], "needs_more_info")
        self.assertTrue(base.name.endswith("20260801-093015"), base.name)

    def test_live_sheet_says_writes_already_happened(self):
        c = Classification("needs_judgment", "Clinical call.", [], [], 0.8, "m")
        stamp = datetime.datetime(2026, 8, 1, 9, 30, 15, tzinfo=datetime.timezone.utc)
        with tempfile.TemporaryDirectory() as d:
            base = run.write_proposals(load_config(), Path(d), stamp,
                                       [(issue(), c, "abc")], live=True)
            live_md = base.with_suffix(".md").read_text()
            base = run.write_proposals(load_config(), Path(d), stamp,
                                       [(issue(), c, "abc")], live=False)
            dry_md = base.with_suffix(".md").read_text()
        self.assertIn("LIVE", live_md)
        self.assertNotIn("Grade in the matching CSV", live_md)
        self.assertIn("dry-run", dry_md)
        self.assertIn("Grade in the matching CSV", dry_md)


class EvalGateTests(unittest.TestCase):
    """The >= 90% gate that authorises go-live, and what it refuses to score."""

    def _harness(self, tmp, cases, labels_returned):
        """A module whose GRADED/CONTEXTS point at tmp, with a stub classifier."""
        module = load_evals_module()
        d = Path(tmp)
        (d / "frozen").mkdir()
        module.GRADED = d / "graded.csv"
        module.CONTEXTS = d / "frozen"
        with open(module.GRADED, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=module.GRADED_COLUMNS)
            w.writeheader()
            for row in cases:
                text = row.pop("_context", f"TICKET: {row['key']}\n")
                if text is not None:
                    (d / "frozen" / f"{row['key']}.txt").write_text(text)
                if row.get("content_hash") == "AUTO":
                    row["content_hash"] = ctx.content_hash(text) if text else ""
                w.writerow(row)
        labels = list(labels_returned)

        class Sequenced:
            def classify(self, text):
                return Classification(labels.pop(0), "r", [], [], 0.9, "m")

        module.Classifier = lambda *a: Sequenced()
        return module

    def _run(self, tmp, cases, labels_returned, gate=0.9):
        module = self._harness(tmp, cases, labels_returned)
        with contextlib.redirect_stdout(io.StringIO()) as out:
            try:
                rc = module.run(gate)
            except SystemExit as e:
                return 1, out.getvalue() + str(e)
        return rc, out.getvalue()

    def _case(self, key, expected="needs_more_info", **kw):
        row = {"key": key, "expected_label": expected, "content_hash": "AUTO", "notes": ""}
        row.update(kw)
        return row

    def test_the_gate_matches_the_agreement_it_reports(self):
        # Exhaustive over small case sets AND over the gate itself, so the
        # boundary is actually reached: with the default 0.9 and fewer than ten
        # cases no ratio ever equals the threshold, and >= vs > is untested.
        for total in range(1, 6):
            for hits in range(total + 1):
                for gate in (0.0, hits / total, 0.9, 1.0):
                    with tempfile.TemporaryDirectory() as tmp:
                        cases = [self._case(f"O3-{i}") for i in range(total)]
                        labels = (["needs_more_info"] * hits
                                  + ["needs_judgment"] * (total - hits))
                        rc, report = self._run(tmp, cases, labels, gate=gate)
                    state = f"{hits}/{total} gate={gate}"
                    self.assertEqual(rc == 0, (hits / total) >= gate, f"{state}\n{report}")
                    self.assertIn(f"agreement: {hits}/{total}", report)

    def test_the_threshold_is_inclusive(self):
        # 9/10 against a 0.9 gate is the real-world boundary this pilot uses.
        with tempfile.TemporaryDirectory() as tmp:
            cases = [self._case(f"O3-{i}") for i in range(10)]
            labels = ["needs_more_info"] * 9 + ["needs_judgment"]
            rc, report = self._run(tmp, cases, labels, gate=0.9)
        self.assertEqual(rc, 0, f"exactly 90% must pass the >= 90% gate\n{report}")

    def test_one_unusable_case_blocks_the_whole_run(self):
        # A rejected case must not simply be dropped, leaving the rest scored:
        # the gate would then be computed over a silently smaller set.
        with tempfile.TemporaryDirectory() as tmp:
            good = self._case("O3-1")
            bad = self._case("O3-2", expected="looks_fine")
            rc, report = self._run(tmp, [good, bad], ["needs_more_info"] * 2)
        self.assertEqual(rc, 1)
        self.assertIn("unusable", report)
        self.assertNotIn("agreement:", report)

    def test_an_edited_frozen_context_is_refused_not_scored(self):
        # Freezing is the whole point of evals/contexts/; a silent edit would
        # move the gate with nobody seeing it.
        with tempfile.TemporaryDirectory() as tmp:
            case = self._case("O3-1")
            case["content_hash"] = ctx.content_hash("TICKET: O3-1\n")
            case["_context"] = "TICKET: O3-1 (edited since grading)\n"
            rc, report = self._run(tmp, [case], ["needs_more_info"])
        self.assertEqual(rc, 1)
        self.assertIn("edited since grading", report)
        self.assertNotIn("agreement:", report)

    def test_a_corrupted_label_is_refused_not_scored(self):
        # Otherwise every case misses and the gate looks like a prompt failure.
        with tempfile.TemporaryDirectory() as tmp:
            rc, report = self._run(tmp, [self._case("O3-1", expected="looks_fine")],
                                   ["needs_more_info"])
        self.assertEqual(rc, 1)
        self.assertIn("is not one of", report)
        self.assertNotIn("agreement:", report)

    def test_a_missing_frozen_context_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            case = self._case("O3-1")
            case["_context"] = None
            rc, report = self._run(tmp, [case], [])
        self.assertEqual(rc, 1)
        self.assertIn("missing", report)

    def test_a_duplicate_key_is_refused_rather_than_double_weighted(self):
        # graded.csv is checked in and hand-edited. A repeated key would count
        # one human judgement twice in the gate's denominator, and silently -
        # the agreement percentage would still look ordinary.
        with tempfile.TemporaryDirectory() as tmp:
            rc, report = self._run(tmp, [self._case("O3-1"), self._case("O3-1")],
                                   ["needs_more_info"] * 2)
        self.assertEqual(rc, 1)
        self.assertIn("appears more than once", report)

    def test_a_sweep_landing_mid_import_cannot_corrupt_the_answer_key(self):
        # The race the read-once freeze exists for: out/contexts/<KEY>.txt is
        # rewritten by every dry-run, and the import used to read it three times
        # - to verify, to copy, and to hash for storage. A sweep between those
        # reads froze text nobody graded, beside a hash that agreed with it, and
        # reported "imported 1". Simulating the rewrite is what makes this bite:
        # without it, reading once and reading three times are indistinguishable.
        module = load_evals_module()
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / "out").mkdir()
            module.GRADED, module.CONTEXTS = d / "graded.csv", d / "frozen"
            graded_text = "TICKET: O3-1\nthe text a human graded\n"
            src = d / "out" / "O3-1.txt"
            src.write_text(graded_text)
            sheet = d / "proposals.csv"
            with open(sheet, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=run.PROPOSAL_COLUMNS)
                w.writeheader()
                w.writerow({"key": "O3-1", "proposed_label": "needs_more_info",
                            "content_hash": ctx.content_hash(graded_text),
                            "source": "api", "grade(ok/wrong)": "ok"})

            real_read = Path.read_text

            def sweep_lands_after_the_first_read(self, *a, **kw):
                out = real_read(self, *a, **kw)
                if self == src:
                    src.write_text("TICKET: O3-1\nrewritten by a later sweep\n")
                return out

            with mock.patch.object(Path, "read_text", sweep_lands_after_the_first_read), \
                 contextlib.redirect_stdout(io.StringIO()):
                module.import_proposals(str(sheet), str(d / "out"))
            with open(module.GRADED) as fh:
                row = next(csv.DictReader(fh))
            frozen = (module.CONTEXTS / "O3-1.txt").read_text()
        self.assertEqual(frozen, graded_text, "froze text that was never graded")
        self.assertEqual(row["content_hash"], ctx.content_hash(graded_text))

    def test_the_frozen_context_matches_the_hash_recorded_beside_it(self):
        # The invariant the import exists to establish. It used to read the
        # source three times - check, copy, hash - and out/contexts is rewritten
        # by every dry-run, so a sweep landing between those reads froze a file
        # and a hash that agreed with each other and with nothing anyone graded.
        module = load_evals_module()
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / "out").mkdir()
            module.GRADED, module.CONTEXTS = d / "graded.csv", d / "frozen"
            text = "TICKET: O3-1\nbody\n"
            (d / "out" / "O3-1.txt").write_text(text)
            sheet = d / "proposals.csv"
            with open(sheet, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=run.PROPOSAL_COLUMNS)
                w.writeheader()
                w.writerow({"key": "O3-1", "proposed_label": "needs_more_info",
                            "content_hash": ctx.content_hash(text), "source": "api",
                            "grade(ok/wrong)": "ok"})
            with contextlib.redirect_stdout(io.StringIO()):
                module.import_proposals(str(sheet), str(d / "out"))
            with open(module.GRADED) as fh:
                row = next(csv.DictReader(fh))
            frozen = (module.CONTEXTS / "O3-1.txt").read_text()
        self.assertEqual(row["content_hash"], ctx.content_hash(frozen))
        self.assertEqual(frozen, text)

    def test_a_non_jira_key_in_the_graded_set_is_refused(self):
        # graded.csv is checked in and hand-edited, and `key` becomes
        # CONTEXTS / f"{key}.txt" - a traversal key reads outside the frozen
        # contexts directory. import_proposals guards this; load_cases did too,
        # but nothing tested it: removing the guard left the whole suite green.
        # Asserting the specific message is what makes this test bite - without
        # the guard the row is still rejected, but for missing its context file.
        with tempfile.TemporaryDirectory() as tmp:
            case = self._case("O3-1")
            case["key"] = "../../../../etc/passwd"
            case["_context"] = None
            rc, report = self._run(tmp, [case], ["needs_more_info"])
        self.assertEqual(rc, 1)
        self.assertIn("not a Jira issue key", report)

    def test_an_unverifiable_case_is_refused_not_scored(self):
        # Previously this warned and then scored the case into the >= 90% gate,
        # contradicting load_cases' own docstring ("refused rather than scored").
        # With no hash there is nothing to prove the frozen text is what a human
        # graded, and the gate it feeds authorises writing to public tickets.
        with tempfile.TemporaryDirectory() as tmp:
            case = self._case("O3-1")
            case["content_hash"] = ""
            rc, report = self._run(tmp, [case], ["needs_more_info"])
        self.assertEqual(rc, 1)
        self.assertIn("no content_hash recorded", report)
        self.assertNotIn("agreement:", report)

    def test_the_result_names_the_model_and_prompt_it_gated(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, report = self._run(tmp, [self._case("O3-1")], ["needs_more_info"])
        cfg = load_config()
        self.assertIn(cfg["claude"]["model"], report)
        self.assertIn(f"prompt {cfg['prompt']['version']}", report)

    def test_a_classifier_error_counts_as_a_miss_not_a_crash(self):
        module = None
        with tempfile.TemporaryDirectory() as tmp:
            module = self._harness(tmp, [self._case("O3-1"), self._case("O3-2")],
                                   ["needs_more_info"])

            class Exploding:
                def __init__(self):
                    self.calls = 0

                def classify(self, text):
                    self.calls += 1
                    if self.calls == 1:
                        raise RuntimeError("529 overloaded")
                    return Classification("needs_more_info", "r", [], [], 0.9, "m")

            module.Classifier = lambda *a: Exploding()
            with contextlib.redirect_stdout(io.StringIO()) as out:
                rc = module.run(0.9)
            report = out.getvalue()
        self.assertEqual(rc, 1)
        self.assertIn("ERROR", report)
        self.assertIn("agreement: 1/2", report)


class ModelComparisonTests(unittest.TestCase):
    """--model runs a comparison that can never inherit the gate's authority."""

    def _harness(self, tmp, cases, labels_by_model):
        """Eval module whose classifier answers per-model and records the model."""
        module = load_evals_module()
        d = Path(tmp)
        (d / "frozen").mkdir()
        module.GRADED = d / "graded.csv"
        module.CONTEXTS = d / "frozen"
        with open(module.GRADED, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=module.GRADED_COLUMNS)
            w.writeheader()
            for key in cases:
                text = f"TICKET: {key}\n"
                (d / "frozen" / f"{key}.txt").write_text(text)
                w.writerow({"key": key, "expected_label": "needs_more_info",
                            "content_hash": ctx.content_hash(text), "notes": ""})
        self.constructed = []
        outer = self

        class PerModel:
            def __init__(self, model, *a):
                self.model = model
                outer.constructed.append(model)
                self.labels = list(labels_by_model.get(model, ["needs_more_info"] * 99))

            def classify(self, text):
                return Classification(self.labels.pop(0), "r", [], [], 0.9, self.model)

        module.Classifier = PerModel
        return module

    def _run(self, tmp, cases, labels_by_model, models=None, gate=0.9):
        module = self._harness(tmp, cases, labels_by_model)
        with contextlib.redirect_stdout(io.StringIO()) as out:
            rc = module.run(gate, models)
        return rc, out.getvalue()

    def test_the_default_invocation_still_gates_the_pinned_model(self):
        # The regression that matters: adding the flag must not change what the
        # gate measures when the flag is absent.
        pinned = load_config()["claude"]["model"]
        with tempfile.TemporaryDirectory() as tmp:
            rc, report = self._run(tmp, ["O3-1"], {})
        self.assertEqual(self.constructed, [pinned])
        self.assertIn("agreement: 1/1", report)
        self.assertNotIn("COMPARISON", report)
        self.assertEqual(rc, 0)

    def test_an_override_scores_the_named_model_not_the_pinned_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            rc, report = self._run(tmp, ["O3-1"], {}, models=["claude-sonnet-5"])
        self.assertEqual(self.constructed, ["claude-sonnet-5"])
        self.assertIn("claude-sonnet-5", report)

    def test_an_override_never_prints_the_gate_verdict(self):
        # The gate's own line is what a reader (or a grep) treats as authorisation.
        # A comparison must not produce it for any agreement level.
        for labels in (["needs_more_info"], ["needs_judgment"]):
            with tempfile.TemporaryDirectory() as tmp:
                _, report = self._run(tmp, ["O3-1"], {"m": labels}, models=["m"])
            self.assertNotIn("agreement: ", report,
                             "a comparison emitted the gate's agreement line")
            for word in ("PASS", "FAIL"):
                self.assertNotIn(word, report,
                                 f"a comparison used the gate's {word} vocabulary")

    def test_an_override_says_it_is_not_the_gate_and_names_the_pinned_model(self):
        pinned = load_config()["claude"]["model"]
        with tempfile.TemporaryDirectory() as tmp:
            _, report = self._run(tmp, ["O3-1"], {}, models=["claude-haiku-4-5"])
        self.assertIn("not the pre-registered gate", report)
        self.assertIn(pinned, report, "the comparison never names what actually gates")

    def test_a_comparisons_exit_code_is_not_a_verdict(self):
        # Zero means "the comparison ran". A model far below the gate must not
        # make it non-zero, or a caller will read the code as a pass/fail.
        with tempfile.TemporaryDirectory() as tmp:
            rc, report = self._run(tmp, ["O3-1", "O3-2"], {"m": ["needs_judgment"] * 2},
                                   models=["m"])
        self.assertEqual(rc, 0)
        self.assertIn("below", report)

    def test_every_requested_model_is_scored_in_one_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, report = self._run(
                tmp, ["O3-1", "O3-2"],
                {"good": ["needs_more_info", "needs_more_info"],
                 "bad": ["needs_judgment", "needs_judgment"]},
                models=["good", "bad"])
        self.assertEqual(self.constructed, ["good", "bad"])
        self.assertIn("2/2", report)
        self.assertIn("0/2", report)

    def test_the_pinned_model_is_marked_when_included_in_a_comparison(self):
        pinned = load_config()["claude"]["model"]
        with tempfile.TemporaryDirectory() as tmp:
            _, report = self._run(tmp, ["O3-1"], {}, models=[pinned, "other"])
        self.assertIn("(pinned)", report)
        self.assertNotIn("was not run", report)

    def test_omitting_the_pinned_model_says_so(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, report = self._run(tmp, ["O3-1"], {}, models=["other"])
        self.assertIn("was not run", report)

    def test_a_comparison_still_refuses_an_unusable_eval_set(self):
        # The integrity checks gate the numbers, not the mode - a comparison
        # measured against an edited context is as meaningless as a gate run.
        with tempfile.TemporaryDirectory() as tmp:
            module = self._harness(tmp, ["O3-1"], {})
            (Path(tmp) / "frozen" / "O3-1.txt").write_text("TICKET: O3-1\nedited\n")
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit):
                module.run(0.9, ["claude-sonnet-5"])


class EvalImportTests(unittest.TestCase):
    """A grade must stay pinned to the context it was actually made against."""

    HEADER = run.PROPOSAL_COLUMNS

    def test_writer_and_reader_agree_on_every_column(self):
        # A real round trip: write a sheet, grade it, import it. A column rename
        # in write_proposals used to leave the reader's tests green while
        # import_proposals silently found no gradable rows.
        module = load_evals_module()
        cfg = load_config()
        c = Classification("needs_more_info", "No repro.", ["repro steps"], [], 0.8, "m")
        text = "TICKET: O3-1\nSUMMARY: Fix the widget\n"
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / "contexts").mkdir()
            (d / "contexts" / "O3-1.txt").write_text(text)
            stamp = datetime.datetime(2026, 8, 1, 9, 0, tzinfo=datetime.timezone.utc)
            base = run.write_proposals(cfg, d, stamp,
                                       [(issue(), c, ctx.content_hash(text))], live=False)
            sheet = base.with_suffix(".csv")
            with open(sheet) as fh:
                rows = list(csv.DictReader(fh))
            rows[0]["grade(ok/wrong)"] = "ok"          # the grader's edit
            with open(sheet, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=run.PROPOSAL_COLUMNS)
                w.writeheader()
                w.writerows(rows)
            module.GRADED = d / "graded.csv"
            module.CONTEXTS = d / "frozen"
            with contextlib.redirect_stdout(io.StringIO()) as log:
                module.import_proposals(str(sheet), str(d / "contexts"))
            with open(module.GRADED) as fh:
                graded = list(csv.DictReader(fh))
        self.assertEqual([r["key"] for r in graded], ["O3-1"], log.getvalue())
        self.assertEqual(graded[0]["expected_label"], "needs_more_info")

    def _import(self, context_text, graded_hash, label="needs_more_info",
                source="api", grade="ok", correct_label="", key="O3-1", header=None):
        module = load_evals_module()
        header = header or self.HEADER
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / "contexts").mkdir()
            if "/" not in key:  # a traversal key has no context to create
                (d / "contexts" / f"{key}.txt").write_text(context_text)
            proposals = d / "proposals.csv"
            row = {"key": key, "proposed_label": label, "confidence": "0.90",
                   "content_hash": graded_hash, "source": source,
                   "grade(ok/wrong)": grade, "correct_label": correct_label}
            with open(proposals, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=header, extrasaction="ignore")
                w.writeheader()
                w.writerow(row)
            # Redirected off the committed evals/ directory.
            module.GRADED = d / "graded.csv"
            module.CONTEXTS = d / "frozen"
            with contextlib.redirect_stdout(io.StringIO()) as log:
                module.import_proposals(str(proposals), str(d / "contexts"))
            with open(module.GRADED) as fh:
                rows = list(csv.DictReader(fh))
        return rows, log.getvalue()

    def test_matching_hash_is_imported(self):
        text = "TICKET: O3-1\n"
        rows, _ = self._import(text, ctx.content_hash(text))
        self.assertEqual([r["key"] for r in rows], ["O3-1"])
        self.assertEqual(rows[0]["expected_label"], "needs_more_info")
        # The hash is recorded, not just checked: without it a later edit to the
        # frozen context cannot be detected at run time.
        self.assertEqual(rows[0]["content_hash"], ctx.content_hash(text))

    def test_context_overwritten_since_grading_is_skipped(self):
        rows, log = self._import("TICKET: O3-1 (edited)\n",
                                 ctx.content_hash("TICKET: O3-1\n"))
        self.assertEqual(rows, [])
        self.assertIn("changed since grading", log)

    def test_ok_graded_replay_is_not_admitted_to_the_eval_set(self):
        # Otherwise the >= 90% gate that authorises go-live would measure the
        # pinned model against a replay classifier's own decision boundary.
        rows, log = self._import("TICKET: O3-1\n", ctx.content_hash("TICKET: O3-1\n"),
                                 source="file")
        self.assertEqual(rows, [])
        self.assertIn("source=file", log)

    def test_absent_source_column_fails_closed(self):
        # This gate authorises go-live, so unknown provenance must not read as
        # proven-api provenance.
        text = "TICKET: O3-1\n"
        rows, log = self._import(text, ctx.content_hash(text),
                                 header=[c for c in run.PROPOSAL_COLUMNS if c != "source"])
        self.assertEqual(rows, [])
        self.assertIn("no source column", log)

    def test_a_key_that_is_not_a_jira_key_is_skipped(self):
        # The key becomes a filesystem path and is stored for later reads.
        rows, log = self._import("x", "", key="../../../../etc/passwd")
        self.assertEqual(rows, [])
        self.assertIn("not a Jira issue key", log)

    def test_wrong_graded_replay_is_admitted_because_the_label_is_human(self):
        rows, _ = self._import("TICKET: O3-1\n", ctx.content_hash("TICKET: O3-1\n"),
                               source="file", grade="wrong", correct_label="needs_judgment")
        self.assertEqual([r["expected_label"] for r in rows], ["needs_judgment"])

    def test_sheet_without_a_hash_column_is_refused(self):
        # Every proposals CSV this pipeline writes carries content_hash, so a
        # sheet lacking it was hand-made or round-tripped through a tool that
        # dropped the column - the same unknown provenance the source gate
        # refuses. Importing it froze whatever out/contexts holds *now* as the
        # answer key for a grade made against text nobody can identify.
        text = "TICKET: O3-1\n"
        rows, printed = self._import(text, "")
        self.assertEqual(rows, [])
        self.assertIn("no content_hash", printed)


class MetricsJira:
    def __init__(self, issues, intro_keys, properties=None):
        self.issues = issues
        self.intro_keys = list(intro_keys)
        self.properties = properties if properties is not None else {}

    def get_property(self, key, prop):
        if self.properties == "raise":
            raise JiraError("429 Too Many Requests")
        return self.properties.get(key)

    def search_keys(self, jql):
        # The intro metric is the only query that filters on labels.
        return list(self.intro_keys) if "labels in" in jql else list(self.issues)

    def issue(self, key, fields, expand_changelog=False):
        return self.issues[key]

    def changelog(self, key, embedded):
        return list((embedded or {}).get("histories", []))


class MetricsWiringTests(unittest.TestCase):
    """main() must feed sla_met and decide the right values.

    Swapping created and labeled_at would make every SLA delta negative, so the
    24h metric would silently report 100% for every cohort forever.
    """

    def _report(self, issue_json, intro_keys=("O3-1",) * 5, properties=None):
        cfg = load_config()
        cfg["metrics"] = dict(cfg["metrics"], pilot_launch="2026-08-01")
        # A live run writes the property alongside the label, so the default
        # fixture supplies one. A bot-labelled ticket *without* one is unknown
        # provenance and now withholds the decision - which is its own test
        # below, not the state every other metric test should be run in.
        if properties is None:
            properties = {"O3-1": {"source": "api", "classifier": "m"}}
        jira = MetricsJira({"O3-1": issue_json}, intro_keys, properties)
        with mock.patch.dict(os.environ, {"TRIAGE_BOT_ACCOUNT_ID": "bot"}), \
             mock.patch.object(metrics, "load_config", lambda: cfg), \
             mock.patch.object(metrics, "jira_from_env", lambda c: jira), \
             contextlib.redirect_stdout(io.StringIO()) as out:
            metrics.main()
        return out.getvalue()

    def _labeled(self, created, labeled_at):
        return issue(labels=[AI[2]], created=created,
                     histories=[label_change("bot", "", AI[2], created=labeled_at)])

    def test_prompt_labeling_reports_full_sla_and_adopts(self):
        report = self._report(self._labeled("2026-08-10T09:00:00.000+0000",
                                            "2026-08-10T11:00:00.000+0000"))
        self.assertIn("sorted within 24h  : 100.0%  [PASS]", report)
        self.assertIn("DECISION: ADOPT", report)

    def test_late_labeling_is_not_silently_credited(self):
        report = self._report(self._labeled("2026-08-10T09:00:00.000+0000",
                                            "2026-08-12T09:00:00.000+0000"))
        self.assertIn("sorted within 24h  : 0.0%  [FAIL]", report)
        self.assertNotIn("ADOPT", report)

    def test_a_replayed_label_suppresses_the_decision(self):
        # The changelog cannot tell a replayed label from a pinned-model one, so
        # deciding over a mixed cohort would measure two systems as one.
        report = self._report(
            self._labeled("2026-08-10T09:00:00.000+0000", "2026-08-10T11:00:00.000+0000"),
            properties={"O3-1": {"source": "file", "classifier": "session-agent"}})
        self.assertIn("were not labelled by the pinned model", report)
        self.assertIn("session-agent", report)
        self.assertIn("NO DECISION", report)
        self.assertNotIn("DECISION: ADOPT", report)

    def test_every_property_read_failing_reports_the_failures(self):
        # A rate-limit window hits every ticket at once. Reporting "the bot has
        # labelled nothing" would be the opposite of the truth, and would discard
        # the diagnostic list that explains it.
        with self.assertRaises(SystemExit) as caught:
            self._report(self._labeled("2026-08-10T09:00:00.000+0000",
                                       "2026-08-10T11:00:00.000+0000"),
                         properties="raise")
        self.assertIn("resolve the failures above", str(caught.exception))

    def test_a_bot_label_with_no_property_withholds_the_decision(self):
        # The property is written after the label and the comment, so a replayed
        # label whose property write raised is public and unattributed. Defaulting
        # that to source=api counted it as pinned-model evidence - failing open on
        # exactly the case the withholding exists for, and silently: the report
        # would print ADOPT over a cohort that mixes two systems.
        report = self._report(self._labeled("2026-08-01T09:00:00.000+0000",
                                            "2026-08-01T10:00:00.000+0000"),
                              properties={})
        self.assertIn("NO DECISION", report)
        self.assertIn("source=absent", report)
        self.assertNotIn("DECISION: ADOPT", report)

    def test_a_malformed_property_is_recorded_not_raised(self):
        # get_property returns whatever the value field holds. A property that is
        # a string or list - hand-set, or written by something else - used to
        # raise AttributeError out of the loop and kill the whole metrics run,
        # discarding every ticket already walked. Per-ticket isolation is the
        # point of the failed list.
        report = self._report(self._labeled("2026-08-01T09:00:00.000+0000",
                                            "2026-08-01T10:00:00.000+0000"),
                              properties={"O3-1": "not-a-dict"})
        self.assertIn("NO DECISION", report)
        self.assertIn("malformed", report)

    def test_an_empty_property_is_unattributed_too(self):
        report = self._report(self._labeled("2026-08-01T09:00:00.000+0000",
                                            "2026-08-01T10:00:00.000+0000"),
                              properties={"O3-1": {}})
        self.assertIn("source=absent", report)

    def test_a_property_without_source_counts_as_api(self):
        # Properties written before `source` existed must not read as replayed.
        report = self._report(
            self._labeled("2026-08-10T09:00:00.000+0000", "2026-08-10T11:00:00.000+0000"),
            properties={"O3-1": {"contentHash": "abc", "prompt": PROMPT_VERSION}})
        self.assertIn("DECISION: ADOPT", report)

    def test_human_relabel_counts_as_removal_and_violation(self):
        report = self._report(issue(
            labels=[AI[0]], created="2026-08-10T09:00:00.000+0000",
            histories=[
                label_change("bot", "", AI[2], created="2026-08-10T10:00:00.000+0000"),
                label_change("u1", AI[2], AI[0], display="Maintainer",
                             created="2026-08-11T09:00:00.000+0000"),
            ]))
        self.assertIn("label removal rate : 1.000  [FAIL]", report)
        self.assertIn("convention adds    : 1", report)
        self.assertIn("Maintainer", report)
        self.assertIn("DECISION: STOP", report)


class DecisionRuleInvariantTests(unittest.TestCase):
    """The pre-registered rule, over its whole input space.

    This rule is committed before launch and decides the pilot. It is a pure
    function of three numbers, so it can be checked exhaustively rather than at
    the handful of points a reviewer happens to pick.
    """

    M = {"sorted_within_24h_pct": 95, "max_label_removal_rate": 0.10,
         "min_intro_outcomes": 5}
    RANK = {"STOP": 0, "EXTEND (two weeks)": 1, "ADOPT": 2}
    PCTS = (0, 50, 90, 94.9, 95, 95.1, 99.9, 100)
    RATES = (0.0, 0.05, 0.099, 0.10, 0.101, 0.15, 0.20, 0.201, 0.5, 1.0)
    INTROS = (0, 1, 4, 5, 6, 50)

    def _grid(self):
        return itertools.product(self.PCTS, self.RATES, self.INTROS)

    def test_every_verdict_is_one_of_the_three(self):
        for pct, rate, intro in self._grid():
            self.assertIn(decide(pct, rate, intro, self.M), self.RANK)

    def test_adopt_exactly_when_all_three_targets_are_met(self):
        for pct, rate, intro in self._grid():
            all_pass = (pct >= 95 and rate <= 0.10 and intro >= 5)
            self.assertEqual(decide(pct, rate, intro, self.M) == "ADOPT", all_pass,
                             f"{pct=} {rate=} {intro=}")

    def test_the_kill_metric_dominates(self):
        # Past double the removal threshold, no amount of throughput or intro
        # output may earn an extension.
        for pct, rate, intro in self._grid():
            if rate > 2 * self.M["max_label_removal_rate"]:
                self.assertEqual(decide(pct, rate, intro, self.M), "STOP",
                                 f"{pct=} {rate=} {intro=}")

    def test_extend_requires_two_passes_and_a_tolerable_removal_rate(self):
        for pct, rate, intro in self._grid():
            if decide(pct, rate, intro, self.M) == "EXTEND (two weeks)":
                passes = sum((pct >= 95, rate <= 0.10, intro >= 5))
                self.assertEqual(passes, 2, f"{pct=} {rate=} {intro=}")
                self.assertLessEqual(rate, 0.20)

    def test_improving_any_metric_never_worsens_the_verdict(self):
        # A rule where doing better scores worse would be a defect no single
        # example is likely to reveal.
        for pct, rate, intro in self._grid():
            base = self.RANK[decide(pct, rate, intro, self.M)]
            for better in (p for p in self.PCTS if p > pct):
                self.assertGreaterEqual(self.RANK[decide(better, rate, intro, self.M)],
                                        base, f"pct {pct}->{better} at {rate=} {intro=}")
            for better in (r for r in self.RATES if r < rate):
                self.assertGreaterEqual(self.RANK[decide(pct, better, intro, self.M)],
                                        base, f"rate {rate}->{better} at {pct=} {intro=}")
            for better in (i for i in self.INTROS if i > intro):
                self.assertGreaterEqual(self.RANK[decide(pct, rate, better, self.M)],
                                        base, f"intro {intro}->{better} at {pct=} {rate=}")


class MetricsReportInvariantTests(unittest.TestCase):
    """The report must never contradict its own verdict."""

    LAUNCH = "2026-08-01"

    def _report(self, tickets, properties=None, intro_keys=("O3-1",) * 5):
        cfg = load_config()
        cfg["metrics"] = dict(cfg["metrics"], pilot_launch=self.LAUNCH)
        jira = MetricsJira(tickets, intro_keys, properties)
        with mock.patch.dict(os.environ, {"TRIAGE_BOT_ACCOUNT_ID": "bot"}), \
             mock.patch.object(metrics, "load_config", lambda: cfg), \
             mock.patch.object(metrics, "jira_from_env", lambda c: jira), \
             contextlib.redirect_stdout(io.StringIO()) as out:
            try:
                rc = metrics.main()
            except SystemExit as e:
                return 1, out.getvalue() + str(e)
        return rc, out.getvalue()

    def _ticket(self, key, created, labeled_at, opted_out=False, human_add=False):
        histories = [label_change("bot", "", AI[2], created=labeled_at)]
        if opted_out:
            histories.append(label_change("u1", AI[2], "", created=labeled_at))
        if human_add:
            histories.append(label_change("u2", "", AI[0], display="Maintainer",
                                          created=labeled_at))
        iss = issue(labels=[AI[2]], created=created, histories=histories)
        iss["key"] = key
        return iss

    def test_no_external_string_can_forge_a_line_in_the_decision_report(self):
        """The report is the pilot's decision artifact, printed one item a line.

        run.py already defends the write side: FileClassifier flattens the
        classifier a classifications file declares, and its comment names this
        exact attack - "a 'DECISION: ADOPT' line in the pilot's own decision
        artifact". Nothing defended the read side. These values come back out
        of a Jira entity property, which anyone with API access to the issue
        can set, and demonstrating it took one property: the report grew a
        DECISION: ADOPT line above its genuine NO DECISION.

        The exit code stayed honest throughout, which is the trap - a reader
        scanning weekly for the verdict line finds one.
        """
        forged = ("agent\nsorted within 24h  : 100.0%  [PASS]  (target >= 95%)\n"
                  "DECISION: ADOPT")
        good = self._ticket("O3-1", "2026-08-10T09:00:00.000+0000",
                            "2026-08-10T10:00:00.000+0000")
        rc, report = self._report({"O3-1": good},
                                  {"O3-1": {"source": "file", "classifier": forged}})
        lines = [l.strip() for l in report.splitlines()]
        self.assertNotIn("DECISION: ADOPT", lines,
                         "a Jira property forged the pilot's verdict line")
        self.assertIn("classifier=agent sorted within 24h", report,
                      "the value must still be reported, on one line")
        self.assertEqual(rc, 1)

    def test_an_exception_body_cannot_forge_a_line_either(self):
        # The same read side, different source: this text quotes an API
        # response body, so a Jira error page or a proxy is enough to reach it
        # without touching the property store at all.
        class Forging(MetricsJira):
            def issue(self, key, fields, expand_changelog=False):
                raise JiraError("500\nDECISION: ADOPT\nsorted within 24h  : 100.0%")

        cfg = load_config()
        cfg["metrics"] = dict(cfg["metrics"], pilot_launch=self.LAUNCH)
        jira = Forging({"O3-1": issue()}, ("O3-1",) * 5, {})
        with mock.patch.dict(os.environ, {"TRIAGE_BOT_ACCOUNT_ID": "bot"}), \
             mock.patch.object(metrics, "load_config", lambda: cfg), \
             mock.patch.object(metrics, "jira_from_env", lambda c: jira), \
             contextlib.redirect_stdout(io.StringIO()) as out:
            with self.assertRaises(SystemExit):
                metrics.main()
        lines = [l.strip() for l in out.getvalue().splitlines()]
        self.assertNotIn("DECISION: ADOPT", lines)

    def test_a_huge_error_body_is_bounded_before_it_reaches_the_report(self):
        # Flattening alone turns a 50KB Jira error page into one 50KB line,
        # which is not an improvement on the report being unreadable. The
        # bound came free with the [:200] this replaced, and nothing asserted
        # it, so it could be dropped silently.
        self.assertLessEqual(len(metrics.one_line("x" * 50000)), 200)
        self.assertLessEqual(len(metrics.one_line("y" * 50000, 60)), 60)
        self.assertEqual(metrics.one_line("a\n\tb  c"), "a b c")

    def test_one_unreadable_ticket_does_not_discard_the_cohort_walked_before_it(self):
        # Coverage found this handler at 0%. The walk is hundreds of requests
        # long, so a single 500 escaping the loop throws away every paid read
        # before it AND the `failed` list explaining why - which is the opposite
        # of the per-ticket isolation the comment above it promises.
        good = self._ticket("O3-1", "2026-08-10T09:00:00.000+0000",
                            "2026-08-10T10:00:00.000+0000")

        class OneBadTicket(MetricsJira):
            def issue(self, key, fields, expand_changelog=False):
                if key == "O3-BAD":
                    raise JiraError("500 Internal Server Error")
                return self.issues[key]

        cfg = load_config()
        cfg["metrics"] = dict(cfg["metrics"], pilot_launch=self.LAUNCH)
        jira = OneBadTicket({"O3-1": good, "O3-BAD": good}, ("O3-1",) * 5,
                            {"O3-1": {"source": "api", "classifier": "m"}})
        with mock.patch.dict(os.environ, {"TRIAGE_BOT_ACCOUNT_ID": "bot"}), \
             mock.patch.object(metrics, "load_config", lambda: cfg), \
             mock.patch.object(metrics, "jira_from_env", lambda c: jira), \
             contextlib.redirect_stdout(io.StringIO()) as out:
            rc = metrics.main()
        report = out.getvalue()
        self.assertIn("tickets labeled    : 1", report, "the readable ticket survived")
        self.assertIn("O3-BAD", report, "the failure must be named, not dropped")
        self.assertIn("NO DECISION", report, "an incomplete cohort cannot decide")
        self.assertEqual(rc, 1)

    def test_an_unparseable_timestamp_fails_its_own_ticket_only(self):
        # The guard added when sla_met moved inside the per-ticket try. Coverage
        # says it had never run: a ticket whose `created` fromisoformat cannot
        # parse would otherwise raise out of the loop and discard the whole walk.
        good = self._ticket("O3-1", "2026-08-10T09:00:00.000+0000",
                            "2026-08-10T10:00:00.000+0000")
        bad = self._ticket("O3-2", "not a timestamp",
                           "2026-08-10T10:00:00.000+0000")
        cfg = load_config()
        cfg["metrics"] = dict(cfg["metrics"], pilot_launch=self.LAUNCH)
        jira = MetricsJira({"O3-1": good, "O3-2": bad}, ("O3-1",) * 5,
                           {"O3-1": {"source": "api", "classifier": "m"},
                            "O3-2": {"source": "api", "classifier": "m"}})
        with mock.patch.dict(os.environ, {"TRIAGE_BOT_ACCOUNT_ID": "bot"}), \
             mock.patch.object(metrics, "load_config", lambda: cfg), \
             mock.patch.object(metrics, "jira_from_env", lambda c: jira), \
             contextlib.redirect_stdout(io.StringIO()) as out:
            rc = metrics.main()
        report = out.getvalue()
        self.assertIn("tickets labeled    : 1", report)
        self.assertIn("computing the 24h SLA", report,
                      "the report must say which step failed, not just that one did")
        self.assertIn("O3-2", report)
        self.assertEqual(rc, 1)

    def test_the_verdict_always_agrees_with_the_printed_pass_flags(self):
        # The rounding that made "94.6%" print as "95%" against a ">= 95%" target
        # is why each line now carries its own PASS/FAIL.
        prompt = "2026-08-10T09:00:00.000+0000", "2026-08-10T11:00:00.000+0000"
        late = "2026-08-10T09:00:00.000+0000", "2026-08-20T09:00:00.000+0000"
        for n_prompt, n_late, opted, intro_n in itertools.product(
                range(0, 4), range(0, 4), (0, 1), (0, 5)):
            if n_prompt + n_late == 0:
                continue
            tickets = {}
            for i in range(n_prompt):
                tickets[f"O3-P{i}"] = self._ticket(f"O3-P{i}", *prompt)
            for i in range(n_late):
                tickets[f"O3-L{i}"] = self._ticket(f"O3-L{i}", *late,
                                                   opted_out=bool(opted and i == 0))
            rc, report = self._report(tickets, intro_keys=("x",) * intro_n)
            state = f"{n_prompt=} {n_late=} {opted=} {intro_n=}"
            if "DECISION:" not in report:
                continue
            verdict = report.split("DECISION:")[1].strip().splitlines()[0].strip()
            passes = report.count("[PASS]")
            # ADOPT iff all three lines said PASS; STOP never with all three.
            self.assertEqual(verdict == "ADOPT", passes == 3, f"{state}\n{report}")
            if verdict == "EXTEND (two weeks)":
                self.assertEqual(passes, 2, f"{state}\n{report}")
            self.assertEqual(rc, 0, state)

    def test_a_blocked_cohort_never_prints_a_decision(self):
        prompt = "2026-08-10T09:00:00.000+0000", "2026-08-10T11:00:00.000+0000"
        tickets = {"O3-1": self._ticket("O3-1", *prompt)}
        for properties in ({"O3-1": {"source": "file", "classifier": "agent"}}, "raise"):
            rc, report = self._report(tickets, properties=properties)
            self.assertEqual(rc, 1, report)
            self.assertNotIn("DECISION: ADOPT", report)
            self.assertNotIn("DECISION: EXTEND", report)
            self.assertNotIn("DECISION: STOP", report)

    def test_both_blockers_are_reported_in_one_pass(self):
        # Otherwise an operator fixes the read failures, re-runs, and only then
        # discovers the cohort was also contaminated.
        prompt = "2026-08-10T09:00:00.000+0000", "2026-08-10T11:00:00.000+0000"
        tickets = {"O3-1": self._ticket("O3-1", *prompt),
                   "O3-2": self._ticket("O3-2", *prompt)}

        class Partial(MetricsJira):
            def get_property(self, key, prop):
                if key == "O3-2":
                    raise JiraError("429 Too Many Requests")
                return {"source": "file", "classifier": "agent"}

        cfg = load_config()
        cfg["metrics"] = dict(cfg["metrics"], pilot_launch=self.LAUNCH)
        jira = Partial(tickets, ("x",) * 5)
        with mock.patch.dict(os.environ, {"TRIAGE_BOT_ACCOUNT_ID": "bot"}), \
             mock.patch.object(metrics, "load_config", lambda: cfg), \
             mock.patch.object(metrics, "jira_from_env", lambda c: jira), \
             contextlib.redirect_stdout(io.StringIO()) as out:
            rc = metrics.main()
        report = out.getvalue()
        self.assertEqual(rc, 1)
        self.assertIn("could not be read", report)
        self.assertIn("not labelled by the pinned model", report)
        self.assertIn("NO DECISION", report)


class LiveRunTests(unittest.TestCase):
    """main() in --live: one label at a time, one comment per label decision."""

    def _run(self, jira, label="needs_more_info", extra_args=(), live=True, out=None):
        classification = Classification(label, "Because.", ["repro steps"], [], 0.9, "m")
        argv = (["--live"] if live else []) + ["--keys", "O3-1", *extra_args]
        with tempfile.TemporaryDirectory() as d:
            d = str(out) if out else d
            # main() reports progress on stdout/stderr; the assertions read the
            # journal and the recorded writes, so keep the test output readable.
            with mock.patch.dict(os.environ, {"TRIAGE_BOT_ACCOUNT_ID": "bot"}), \
                 mock.patch.object(run, "jira_from_env", lambda cfg: jira), \
                 mock.patch.object(run, "Classifier", lambda *a: StubClassifier(classification)), \
                 contextlib.redirect_stdout(io.StringIO()), \
                 contextlib.redirect_stderr(io.StringIO()):
                run.main(argv, out=Path(d))
            journal = (Path(d) / "journal.jsonl").read_text().splitlines()
        return json.loads(journal[-1])

    def test_fresh_ticket_labels_then_comments_then_records_hash(self):
        jira = RecordingJira({"O3-1": issue()})
        row = self._run(jira)
        self.assertEqual([w[0] for w in jira.writes], ["labels", "comment", "property"])
        self.assertEqual(row["action"], "labeled")

    def test_label_flip_removes_stale_and_comments_once(self):
        jira = RecordingJira({"O3-1": issue(labels=[AI[0]])})
        self._run(jira)
        self.assertEqual([w[0] for w in jira.writes], ["labels", "comment", "property"])
        _, _, added, removed = jira.writes[0]
        self.assertEqual((added, removed), ((AI[2],), (AI[0],)))

    def test_rerun_with_matching_hash_writes_nothing(self):
        jira = RecordingJira({"O3-1": issue(labels=[AI[2]])})
        self._run(jira)
        jira.writes.clear()
        row = self._run(jira)
        self.assertEqual(jira.writes, [])
        self.assertEqual(row["action"], "skip-already-triaged")

    def test_edited_ticket_refreshes_without_a_second_comment(self):
        jira = RecordingJira({"O3-1": issue(labels=[AI[2]])})
        self._run(jira)
        jira.issues["O3-1"] = issue(labels=[AI[2]], description="Now with repro steps.")
        jira.writes.clear()
        row = self._run(jira)
        self.assertNotIn("comment", [w[0] for w in jira.writes])
        self.assertEqual(row["action"], "refreshed")

    def test_prompt_bump_reclassifies_an_otherwise_unchanged_ticket(self):
        # Otherwise a mid-pilot prompt fix would apply only to newly-swept
        # tickets, leaving the cohort graded under two prompt versions.
        jira = RecordingJira({"O3-1": issue(labels=[AI[2]])})
        self._run(jira)
        self.assertEqual(jira.properties["O3-1"]["prompt"], PROMPT_VERSION)
        jira.properties["O3-1"]["prompt"] = "v0"
        jira.writes.clear()
        row = self._run(jira)
        self.assertEqual(row["action"], "refreshed")
        self.assertEqual([w[0] for w in jira.writes], ["property"])

    def test_ticket_transitioned_out_of_scope_is_untouched(self):
        # The race: it matched status = "To Do" in the sweep, then moved before
        # this fetch. Labelling it invites a removal, which is a permanent
        # opt-out and counts against the removal-rate kill metric.
        jira = RecordingJira({"O3-1": issue(status={"name": "In Progress"})})
        row = self._run(jira)
        self.assertEqual(jira.writes, [])
        self.assertEqual(row["action"], "skip-out-of-scope")
        self.assertEqual(row["status"], "In Progress")

    def test_scope_is_not_enforced_in_dry_run(self):
        # Dry-run is the inspection and grading tool; --keys must still work on
        # any ticket.
        jira = RecordingJira({"O3-1": issue(status={"name": "In Progress"})})
        row = self._run(jira, live=False)
        self.assertEqual(jira.writes, [])
        self.assertEqual(row["action"], "proposed")

    def test_failed_write_is_absent_from_the_live_audit_sheet(self):
        # The sheet is headed "labels and comments applied"; a ticket whose write
        # raised must not be listed under that claim.
        class Failing(RecordingJira):
            def add_comment(self, key, body):
                raise JiraError("PUT /issue/O3-1 -> 403: Edit Issues missing")

        jira = Failing({"O3-1": issue()})
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            self._run(jira, out=out)
            sheets = list(out.glob("proposals-*.md"))
        self.assertEqual(sheets, [], "no sheet should be written for a failed-only run")

    def test_sweep_stops_after_consecutive_failures(self):
        # A systemic fault would otherwise re-spend a paid call on every ticket
        # of every four-hourly sweep, silently.
        class Failing(RecordingJira):
            def __init__(self, issues):
                super().__init__(issues)
                self.fetches = 0

            def issue(self, key, fields, expand_changelog=False):
                self.fetches += 1
                raise JiraError("500 Internal Server Error")

        keys = [f"O3-{i}" for i in range(20)]
        jira = Failing({k: issue() for k in keys})
        with tempfile.TemporaryDirectory() as d:
            with mock.patch.dict(os.environ, {"TRIAGE_BOT_ACCOUNT_ID": "bot"}), \
                 mock.patch.object(run, "jira_from_env", lambda cfg: jira), \
                 mock.patch.object(run, "Classifier", lambda *a: StubClassifier(None)), \
                 contextlib.redirect_stdout(io.StringIO()), \
                 contextlib.redirect_stderr(io.StringIO()):
                rc = run.main(["--live"], out=Path(d))
        self.assertEqual(rc, 1, "a failed sweep must not exit 0")
        self.assertEqual(jira.fetches, run.CONSECUTIVE_ERROR_LIMIT)

    def _sweep(self, jira, classification, argv, out):
        with mock.patch.dict(os.environ, {"TRIAGE_BOT_ACCOUNT_ID": "bot"}), \
             mock.patch.object(run, "jira_from_env", lambda cfg: jira), \
             mock.patch.object(run, "Classifier",
                               lambda *a: StubClassifier(classification)), \
             contextlib.redirect_stdout(io.StringIO()), \
             contextlib.redirect_stderr(io.StringIO()):
            rc = run.main(list(argv), out=out)
        rows = [json.loads(l) for l in (out / "journal.jsonl").read_text().splitlines()]
        return rc, rows

    def test_a_refusal_is_never_written_as_a_label(self):
        # A refusal carries label="" by construction. Falling through to the
        # write path turns that into cfg["labels"][""] - and in a dry run, into
        # a blank-labelled row on the page headed "what the triage pilot wrote",
        # which is the page Dennis and Veronica review.
        refused = Classification("", "", [], [], 0.0, "m", refused=True)
        jira = RecordingJira({"O3-1": issue()})
        with tempfile.TemporaryDirectory() as d:
            rc, rows = self._sweep(jira, refused, ["--live", "--keys", "O3-1"], Path(d))
        self.assertEqual(rows[-1]["action"], "error-refusal")
        self.assertEqual(jira.writes, [], "a refusal must touch nothing in Jira")

    def test_consecutive_refusals_trip_the_breaker_like_any_other_error(self):
        # A refusal costs a paid call exactly like a failure does, and a prompt
        # that trips the safety classifier trips it on every ticket. Excusing
        # refusals from the breaker burns the whole cohort, then does it again
        # every four hours - the precise spend the breaker exists to cap.
        refused = Classification("", "", [], [], 0.0, "m", refused=True)
        keys = [f"O3-{i}" for i in range(20)]
        jira = RecordingJira({k: issue() for k in keys})
        with tempfile.TemporaryDirectory() as d:
            rc, rows = self._sweep(jira, refused, ["--live"], Path(d))
        self.assertEqual(len(rows), run.CONSECUTIVE_ERROR_LIMIT,
                         "the sweep must stop, not refuse its way through the cohort")
        self.assertEqual(rc, 1)

    def test_a_comment_that_landed_is_reported_even_if_bookkeeping_fails(self):
        # Ordering, not presence: the append must happen BEFORE set_property.
        # The label and comment are already public at that point, so dropping
        # the ticket when only the internal property write fails hides a real
        # comment from the report that claims to list them - and set_property
        # is the permission preflight probes precisely because it is the one
        # most likely to be missing.
        class NoProperties(RecordingJira):
            def set_property(self, key, prop, value):
                raise JiraError("403: no entity-property permission")

        c = Classification("automation_candidate", "Clear spec.", [], ["run it"], 0.9, "m")
        jira = NoProperties({"O3-1": issue()})
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            rc, rows = self._sweep(jira, c, ["--live", "--keys", "O3-1"], out)
            report = next(out.glob("proposals-*.html")).read_text()
        self.assertEqual(rows[-1]["action"], "error", "the failure is still reported")
        self.assertIn("O3-1", report,
                      "the comment is on the ticket, so it must be in the report")

    def test_a_limited_gather_does_not_claim_a_complete_manifest(self):
        # `complete` is what the apply step trusts to decide the cohort is fully
        # described. A --limit run covers a prefix of it; claiming complete
        # means every ticket past the limit is never classified at all, and
        # nothing anywhere says so.
        keys = [f"O3-{i}" for i in range(5)]
        jira = RecordingJira({k: issue() for k in keys})
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            self._sweep(jira, None, ["--no-classify", "--limit", "2"], out)
            manifest = json.loads((out / "manifest.json").read_text())
        self.assertEqual(len(manifest["tickets"]), 2)
        self.assertFalse(manifest["complete"],
                         "a manifest covering 2 of 5 tickets is not complete")

    def test_an_opt_out_survives_every_later_sweep(self):
        """The pilot's central promise, run as a sequence rather than a fixture.

        Every existing opt-out test is either inspect() in isolation or a single
        sweep against a hand-written changelog. Neither can show that the
        pipeline PRODUCES a history its own reader then interprets as an
        opt-out: sweep one's add and the maintainer's removal are a pair, and
        only a real sequence puts both in the log the way Jira would.

        The promise is not "skip it once". The comment posted on every ticket
        says removing the label opts the ticket out, so it has to outlast a
        content edit, a prompt bump and an explicit --force - each of which
        exists precisely to make the pipeline reconsider a ticket.
        """
        c = Classification("needs_judgment", "A clinical call.", [], [], 0.8, "m")
        label = load_config()["labels"]["needs_judgment"]
        jira = LivingJira({"O3-1": issue()})

        def sweep(*extra):
            with tempfile.TemporaryDirectory() as d:
                return self._sweep(jira, c, ["--live", "--keys", "O3-1", *extra], Path(d))

        rc, rows = sweep()
        self.assertEqual(rows[-1]["action"], "labeled")
        self.assertIn(("labels", "O3-1", (label,), ()), jira.writes)

        jira.human_removes("O3-1", label)
        writes_at_opt_out = len(jira.writes)

        # 1. plain re-sweep 2. after a content edit 3. after a prompt bump
        # 4. under --force. Each is a different reason the pipeline would
        # normally act, and none may override a human's opt-out.
        rc, rows = sweep()
        self.assertEqual(rows[-1]["action"], "skip-opted-out")
        self.assertEqual(rows[-1]["by"], "A Maintainer")

        jira.issues["O3-1"]["fields"]["summary"] = "Fix the widget, urgently"
        rc, rows = sweep()
        self.assertEqual(rows[-1]["action"], "skip-opted-out",
                         "an edited ticket is still an opted-out ticket")

        jira.properties["O3-1"] = dict(jira.properties.get("O3-1", {}), prompt="v0")
        rc, rows = sweep()
        self.assertEqual(rows[-1]["action"], "skip-opted-out",
                         "a prompt bump must not resurrect an opted-out ticket")

        rc, rows = sweep("--force")
        self.assertEqual(rows[-1]["action"], "skip-opted-out", "--force must not either")

        self.assertEqual(len(jira.writes), writes_at_opt_out,
                         f"the pilot wrote to an opted-out ticket: "
                         f"{jira.writes[writes_at_opt_out:]}")

    def test_a_second_sweep_of_an_unchanged_ticket_stays_silent(self):
        # The counterpart: the living changelog must not make the pipeline
        # paranoid about its own history.
        c = Classification("needs_judgment", "A clinical call.", [], [], 0.8, "m")
        jira = LivingJira({"O3-1": issue()})
        with tempfile.TemporaryDirectory() as d:
            self._sweep(jira, c, ["--live", "--keys", "O3-1"], Path(d))
        after_first = len(jira.writes)
        with tempfile.TemporaryDirectory() as d:
            rc, rows = self._sweep(jira, c, ["--live", "--keys", "O3-1"], Path(d))
        self.assertEqual(rows[-1]["action"], "skip-already-triaged")
        self.assertEqual(len(jira.writes), after_first, "a quiet re-run wrote something")

    def test_the_bots_own_label_flip_is_not_read_as_an_opt_out(self):
        """Three sweeps, because that is the shortest sequence that can fail.

        This is the cohort-wide failure bot_identity_error warns about, and
        working out how to reach it corrected me: a misread label ADD is only a
        convention violation, so the bot's first sweep cannot trigger it. The
        opt-out comes from a misread REMOVAL, and the bot only removes a label
        when it flips one - which needs an edit between two sweeps. Verified by
        simulating a wrong TRIAGE_BOT_ACCOUNT_ID: fourteen tests notice, and
        before this one, not a single sweep-level test was among them.

        If it ever regresses, the bot re-classifies a ticket, reads its own
        removal as a maintainer's opt-out, and permanently excludes it - across
        the cohort, silently, while the removal metric reports a kill.
        """
        first = Classification("needs_more_info", "No steps.", ["repro steps"], [], 0.8, "m")
        second = Classification("needs_judgment", "A clinical call.", [], [], 0.8, "m")
        cfg = load_config()
        jira = LivingJira({"O3-1": issue()})

        def sweep(c):
            with tempfile.TemporaryDirectory() as d:
                return self._sweep(jira, c, ["--live", "--keys", "O3-1"], Path(d))

        rc, rows = sweep(first)
        self.assertEqual(rows[-1]["action"], "labeled")

        # An edit changes the content hash, so the next sweep reclassifies -
        # and this time to a different label, so the bot removes its own.
        jira.issues["O3-1"]["fields"]["description"] = "Steps: open the ward view."
        rc, rows = sweep(second)
        self.assertEqual(rows[-1]["action"], "labeled")
        self.assertIn(("labels", "O3-1", (cfg["labels"]["needs_judgment"],),
                       (cfg["labels"]["needs_more_info"],)), jira.writes,
                      "the flip must actually remove the old label")

        # The removal is now in the changelog, authored by the bot. Read as a
        # human's, this ticket is opted out forever.
        rc, rows = sweep(second)
        self.assertEqual(rows[-1]["action"], "skip-already-triaged",
                         "the bot's own flip was read as a maintainer opt-out")

    def test_a_hand_set_entity_property_cannot_halt_the_pilot(self):
        """Availability, not correctness: the sweep used to stop for good.

        The property is our bookkeeping but it lives in Jira, so anyone with
        API access to an issue can write anything under the key - which is
        published in this repo. A non-dict raised AttributeError before
        classification, so the ticket errored; five of them tripped the
        consecutive-error breaker. scope_jql orders by created ASC, so the same
        five are reached first on every run: every sweep aborts, forever, and
        the only evidence is a journal line.

        metrics.py already handled this read defensively - "hand-set, or
        written by something else" - so the thought had occurred once, on the
        other reader of the same property.
        """
        class HandSet(RecordingJira):
            def get_property(self, key, prop):
                return ["not", "an", "object"]

        c = Classification("needs_judgment", "A clinical call.", [], [], 0.8, "m")
        jira = HandSet({f"O3-{i}": issue() for i in range(8)})
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            rc, rows = self._sweep(jira, c, ["--live"], out)
        self.assertEqual(len(rows), 8, "the sweep stopped short of the cohort")
        self.assertEqual([r["action"] for r in rows], ["labeled"] * 8,
                         "a malformed property must not error the ticket")
        self.assertEqual(rc, 0)

    def test_a_malformed_property_is_repaired_rather_than_left(self):
        # Treating it as absent is only safe because the ticket is then
        # re-classified and the property rewritten - one classification, and
        # the ticket is healthy again. If it were left in place the warning
        # would repeat on every sweep for the life of the pilot.
        class HandSet(RecordingJira):
            def get_property(self, key, prop):
                return self.properties.get(key, "a bare string")

        c = Classification("needs_judgment", "A clinical call.", [], [], 0.8, "m")
        jira = HandSet({"O3-1": issue()})
        with tempfile.TemporaryDirectory() as d:
            self._sweep(jira, c, ["--live", "--keys", "O3-1"], Path(d))
        self.assertIsInstance(jira.properties.get("O3-1"), dict,
                              "the malformed property was not overwritten")
        with tempfile.TemporaryDirectory() as d:
            rc, rows = self._sweep(jira, c, ["--live", "--keys", "O3-1"], Path(d))
        self.assertEqual(rows[-1]["action"], "skip-already-triaged",
                         "the repaired property must settle on the next sweep")

    def test_the_journal_describes_what_was_actually_written(self):
        """The whole loop's state space, checked against the audit trail.

        Individual branches are tested and plan_ticket's decision chain was
        enumerated, but the loop composes pieces added in different passes -
        the open-PR skip, the property guard, the write ordering, the breaker -
        and the skill's rule is to re-derive the merged result rather than
        trust the reviews that approved each piece alone.

        The invariant nothing checked is fidelity: the journal is what Dennis
        and Veronica read to see what the bot did, so an action that disagrees
        with the writes is an audit trail that lies. "labeled" must mean a
        comment was posted, "refreshed" must mean one was not, and any skip
        must mean nothing was written at all.
        """
        cfg = load_config()
        chosen = cfg["labels"]["needs_judgment"]
        other = cfg["labels"]["needs_more_info"]
        c = Classification("needs_judgment", "A clinical call.", [], [], 0.8, "m")
        fresh = ctx.content_hash(ctx.assemble(StubJira(), issue(), None, []))
        properties = {
            "absent": None,
            "empty": {},
            "current": {"contentHash": fresh, "prompt": cfg["prompt"]["version"],
                        "source": "api"},
            "stale-hash": {"contentHash": "old", "prompt": cfg["prompt"]["version"],
                           "source": "api"},
            "stale-prompt": {"contentHash": fresh, "prompt": "v0", "source": "api"},
            "replayed": {"contentHash": fresh, "prompt": cfg["prompt"]["version"],
                         "source": "file"},
            "malformed": ["not", "an", "object"],
        }
        seen = set()
        for labels in ([], [chosen], [other]):
            for pname, prop in properties.items():
                for opted_out in (False, True):
                    hist = ([label_change("u1", chosen, "", display="M")]
                            if opted_out else [])

                    class Fixed(RecordingJira):
                        def get_property(self, key, name):
                            return prop

                    jira = Fixed({"O3-1": issue(labels=list(labels), histories=hist)})
                    with tempfile.TemporaryDirectory() as d:
                        _, rows = self._sweep(jira, c, ["--live", "--keys", "O3-1"],
                                              Path(d))
                    action = rows[-1]["action"]
                    kinds = [w[0] for w in jira.writes]
                    where = f"labels={labels} property={pname} opted_out={opted_out}"
                    seen.add(action)

                    if opted_out:
                        self.assertEqual(action, "skip-opted-out", where)
                    if action.startswith("skip"):
                        self.assertEqual(jira.writes, [],
                                         f"journal says {action} but wrote: {where}")
                    if action == "labeled":
                        self.assertEqual(kinds.count("comment"), 1,
                                         f"journal says labeled without a comment: {where}")
                    if action == "refreshed":
                        self.assertEqual(kinds.count("comment"), 0,
                                         f"journal says refreshed but commented: {where}")
                    if action != "error":
                        self.assertNotIn("error", rows[-1], where)

        self.assertEqual(seen, {"labeled", "refreshed", "skip-already-triaged",
                                "skip-opted-out"},
                         f"the enumeration missed an outcome; reached {sorted(seen)}")

    def test_live_without_credentials_refuses_to_start(self):
        # The guard against a --live run that would sweep anonymously: it cannot
        # write, so every ticket fails, but it would fail them five at a time on
        # a schedule while looking like a configured pilot.
        jira = RecordingJira({"O3-1": issue()})
        jira.authenticated = False
        with tempfile.TemporaryDirectory() as d:
            with mock.patch.dict(os.environ, {}, clear=True), \
                 mock.patch.object(run, "jira_from_env", lambda cfg: jira), \
                 contextlib.redirect_stdout(io.StringIO()), \
                 contextlib.redirect_stderr(io.StringIO()), \
                 self.assertRaises(SystemExit) as caught:
                run.main(["--live", "--keys", "O3-1"], out=Path(d))
        self.assertIn("TRIAGE_BOT_ACCOUNT_ID", str(caught.exception))

    def test_a_rejected_dev_panel_clause_sweeps_without_it_rather_than_dying(self):
        # Jira rejects development[] where the GitHub app is not installed. The
        # fallback matches on the error text, and if that match is wrong the
        # sweep dies on its first call instead of degrading - so the condition
        # itself needs exercising against a realistic message.
        cfg = load_config()
        clause = cfg["jira"]["dev_panel_clause"]

        class ClauseRejected(RecordingJira):
            def __init__(self, issues):
                super().__init__(issues)
                self.queries = []

            def search_keys(self, jql):
                self.queries.append(jql)
                if clause in jql:
                    raise JiraError("400: Field 'development' does not exist or you "
                                    "do not have permission to view it.")
                return list(self.issues)

        jira = ClauseRejected({"O3-1": issue()})
        with tempfile.TemporaryDirectory() as d:
            err = io.StringIO()
            with mock.patch.dict(os.environ, {"TRIAGE_BOT_ACCOUNT_ID": "bot"}), \
                 mock.patch.object(run, "jira_from_env", lambda cfg: jira), \
                 mock.patch.object(run, "Classifier", lambda *a: StubClassifier(None)), \
                 contextlib.redirect_stdout(io.StringIO()), \
                 contextlib.redirect_stderr(err):
                run.main(["--no-classify"], out=Path(d))
        self.assertEqual(len(jira.queries), 2, "it must retry without the clause")
        self.assertNotIn(clause, jira.queries[1])
        self.assertIn("development[] JQL clause rejected", err.getvalue(),
                      "degrading to an unfiltered sweep must be said out loud")

    def test_an_unrelated_search_failure_is_not_swallowed_as_a_clause_problem(self):
        # The fallback is scoped to the clause. A 500 or an auth failure must
        # still abort, or a broken Jira produces a confident empty cohort.
        class Broken(RecordingJira):
            def search_keys(self, jql):
                raise JiraError("500 Internal Server Error")

        jira = Broken({"O3-1": issue()})
        with tempfile.TemporaryDirectory() as d:
            with mock.patch.dict(os.environ, {"TRIAGE_BOT_ACCOUNT_ID": "bot"}), \
                 mock.patch.object(run, "jira_from_env", lambda cfg: jira), \
                 mock.patch.object(run, "Classifier", lambda *a: StubClassifier(None)), \
                 contextlib.redirect_stdout(io.StringIO()), \
                 contextlib.redirect_stderr(io.StringIO()), \
                 self.assertRaises(JiraError):
                run.main(["--no-classify"], out=Path(d))

    def test_all_tickets_failing_returns_nonzero(self):
        class Failing(RecordingJira):
            def issue(self, key, fields, expand_changelog=False):
                raise JiraError("boom")

        jira = Failing({"O3-1": issue()})
        with tempfile.TemporaryDirectory() as d:
            with mock.patch.dict(os.environ, {"TRIAGE_BOT_ACCOUNT_ID": "bot"}), \
                 mock.patch.object(run, "jira_from_env", lambda cfg: jira), \
                 mock.patch.object(run, "Classifier", lambda *a: StubClassifier(None)), \
                 contextlib.redirect_stdout(io.StringIO()), \
                 contextlib.redirect_stderr(io.StringIO()):
                rc = run.main(["--live", "--keys", "O3-1"], out=Path(d))
        self.assertEqual(rc, 1)

    def test_replayed_classification_writes_to_jira_without_a_claude_call(self):
        # The point of agent-classifier mode: a live run with no Anthropic
        # credential at all. Classifier is patched to explode if constructed.
        jira = RecordingJira({"O3-1": issue()})
        text = ctx.assemble(StubJira(), jira.issues["O3-1"], None, ["bot"])

        def explode(*a):
            raise AssertionError("the API classifier must not be constructed")

        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "c.json"
            path.write_text(json.dumps({
                "prompt_version": PROMPT_VERSION, "classifier": "session-agent",
                "classifications": {"O3-1": dict(GOOD, content_hash=ctx.content_hash(text))},
            }))
            with mock.patch.dict(os.environ, {"TRIAGE_BOT_ACCOUNT_ID": "bot"}), \
                 mock.patch.object(run, "jira_from_env", lambda cfg: jira), \
                 mock.patch.object(run, "Classifier", explode), \
                 contextlib.redirect_stdout(io.StringIO()), \
                 contextlib.redirect_stderr(io.StringIO()):
                rc = run.main(["--live", "--keys", "O3-1",
                               "--classifications", str(path)], out=Path(d))
        self.assertEqual(rc, 0)
        self.assertEqual([w[0] for w in jira.writes], ["labels", "comment", "property"])
        # Attribution recorded, so an audit can separate replayed labels from
        # pinned-model ones - they are not the same experiment.
        self.assertEqual(jira.properties["O3-1"]["classifier"], "session-agent")

    def test_hostile_rationale_from_a_file_is_escaped_in_the_comment(self):
        # The file path has no server-side constraint at all, so this is the
        # least trusted route into a public Jira comment.
        jira = RecordingJira({"O3-1": issue()})
        text = ctx.assemble(StubJira(), jira.issues["O3-1"], None, ["bot"])
        hostile = dict(
            GOOD,
            content_hash=ctx.content_hash(text),
            rationale="Ping [~accountid:712020:abc] and see !https://attacker.example/p.png!",
            missing_info=["ask [~accountid:712020:def]"],
        )
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "c.json"
            path.write_text(json.dumps({
                "prompt_version": PROMPT_VERSION, "classifier": "agent",
                "classifications": {"O3-1": hostile},
            }))
            with mock.patch.dict(os.environ, {"TRIAGE_BOT_ACCOUNT_ID": "bot"}), \
                 mock.patch.object(run, "jira_from_env", lambda cfg: jira), \
                 mock.patch.object(run, "Classifier", lambda *a: None), \
                 contextlib.redirect_stdout(io.StringIO()), \
                 contextlib.redirect_stderr(io.StringIO()):
                run.main(["--live", "--keys", "O3-1", "--classifications", str(path)],
                         out=Path(d))
        body = next(w[2] for w in jira.writes if w[0] == "comment")
        self.assertNotIn("[~", body.replace("\\[", ""))
        self.assertNotIn("!", body.replace("\\!", ""))

    def test_partial_batch_applies_every_covered_ticket(self):
        # The documented workflow gathers a subset and applies over the sweep, so
        # misses are routine. Counting them as faults aborted the sweep and
        # dropped good classifications - here the covered tickets are LAST, the
        # ordering that previously applied nothing at all.
        keys = [f"O3-{i}" for i in range(1, 11)]
        jira = RecordingJira({k: issue() for k in keys})
        for k in keys:
            jira.issues[k]["key"] = k
        covered = keys[-2:]
        entries = {}
        for k in covered:
            text = ctx.assemble(StubJira(), jira.issues[k], None, ["bot"])
            entries[k] = dict(GOOD, content_hash=ctx.content_hash(text))
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "c.json"
            path.write_text(json.dumps({
                "prompt_version": PROMPT_VERSION, "classifier": "agent",
                "classifications": entries,
            }))
            with mock.patch.dict(os.environ, {"TRIAGE_BOT_ACCOUNT_ID": "bot"}), \
                 mock.patch.object(run, "jira_from_env", lambda cfg: jira), \
                 mock.patch.object(run, "Classifier", lambda *a: None), \
                 contextlib.redirect_stdout(io.StringIO()), \
                 contextlib.redirect_stderr(io.StringIO()):
                rc = run.main(["--live", "--keys", ",".join(keys),
                               "--classifications", str(path)], out=Path(d))
            rows = [json.loads(x) for x in
                    (Path(d) / "journal.jsonl").read_text().splitlines()]
        self.assertEqual(rc, 0, "uncovered tickets are skips, not failures")
        labelled = sorted(w[1] for w in jira.writes if w[0] == "labels")
        self.assertEqual(labelled, sorted(covered))
        self.assertEqual(len(rows), len(keys), "the sweep must not abort early")
        self.assertEqual(sum(r["action"] == "skip-unclassified" for r in rows), 8)

    def test_the_two_paths_do_not_flip_the_label_back_and_forth(self):
        # The idempotency rule is asymmetric on purpose. If a file run also
        # re-did api-labelled tickets, the documented batch workflow plus the
        # 4-hourly sweep would swap the label and comment to every watcher
        # forever, with nothing about the ticket having changed.
        jira = RecordingJira({"O3-1": issue()})
        text = ctx.assemble(StubJira(), jira.issues["O3-1"], None, ["bot"])
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "c.json"
            path.write_text(json.dumps({
                "prompt_version": PROMPT_VERSION, "classifier": "agent",
                "classifications": {"O3-1": dict(GOOD, content_hash=ctx.content_hash(text),
                                                 label="needs_judgment", missing_info=[])},
            }))
            self._run(jira)                      # api labels it needs_more_info
            jira.writes.clear()
            self._replay(jira, path, Path(d))    # file run must leave it alone
            self.assertEqual(jira.writes, [], "a file run must not re-do an api label")
            row = self._run(jira)                # api run must also leave it alone
        self.assertEqual(row["action"], "skip-already-triaged")
        self.assertEqual(jira.writes, [])

    def _replay(self, jira, path, out):
        with mock.patch.dict(os.environ, {"TRIAGE_BOT_ACCOUNT_ID": "bot"}), \
             mock.patch.object(run, "jira_from_env", lambda cfg: jira), \
             mock.patch.object(run, "Classifier", lambda *a: None), \
             contextlib.redirect_stdout(io.StringIO()), \
             contextlib.redirect_stderr(io.StringIO()):
            return run.main(["--live", "--keys", "O3-1", "--classifications", str(path)],
                            out=out)

    def test_replayed_label_does_not_block_the_pinned_model(self):
        # Without comparing source, one replay run would pin those tickets for
        # the rest of the pilot: the API pipeline would skip them forever.
        jira = RecordingJira({"O3-1": issue()})
        text = ctx.assemble(StubJira(), jira.issues["O3-1"], None, ["bot"])
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "c.json"
            path.write_text(json.dumps({
                "prompt_version": PROMPT_VERSION, "classifier": "agent",
                "classifications": {"O3-1": dict(GOOD, content_hash=ctx.content_hash(text))},
            }))
            with mock.patch.dict(os.environ, {"TRIAGE_BOT_ACCOUNT_ID": "bot"}), \
                 mock.patch.object(run, "jira_from_env", lambda cfg: jira), \
                 mock.patch.object(run, "Classifier", lambda *a: None), \
                 contextlib.redirect_stdout(io.StringIO()), \
                 contextlib.redirect_stderr(io.StringIO()):
                run.main(["--live", "--keys", "O3-1", "--classifications", str(path)],
                         out=Path(d))
        self.assertEqual(jira.properties["O3-1"]["source"], "file")
        jira.writes.clear()
        # Now the pinned-model path over the same unchanged ticket.
        row = self._run(jira)
        self.assertEqual(row["action"], "refreshed")
        self.assertEqual(jira.properties["O3-1"]["source"], "api")

    def test_opted_out_ticket_is_untouched_even_with_force(self):
        jira = RecordingJira({"O3-1": issue(histories=[label_change("u1", AI[2], "")])})
        row = self._run(jira, extra_args=["--force"])
        self.assertEqual(jira.writes, [])
        self.assertEqual(row["action"], "skip-opted-out")


class DotenvTests(unittest.TestCase):
    def test_env_loaded_without_overriding(self):
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            env = Path(d) / ".env"
            env.write_text("# comment\nTRIAGE_TEST_A='one'\nTRIAGE_TEST_B=two\n")
            os.environ["TRIAGE_TEST_B"] = "preset"
            try:
                _load_dotenv(env)
                self.assertEqual(os.environ["TRIAGE_TEST_A"], "one")
                self.assertEqual(os.environ["TRIAGE_TEST_B"], "preset")
            finally:
                os.environ.pop("TRIAGE_TEST_A", None)
                os.environ.pop("TRIAGE_TEST_B", None)


class FakeBlock:
    def __init__(self, text, type="text"):
        self.type = type
        self.text = text


class FakeResponse:
    def __init__(self, stop_reason, content=(), model="claude-opus-5"):
        self.stop_reason = stop_reason
        self.content = list(content)
        self.model = model


class StubAnthropic:
    """Stands in for the Anthropic client; records the request it was given."""

    def __init__(self, response):
        self.response = response
        self.kwargs: dict = {}
        self.beta = self
        self.messages = self

    def create(self, **kwargs):
        self.kwargs = kwargs
        return self.response


class ClampTests(unittest.TestCase):
    """The API path clamps; discarding a correct label is the costly choice."""

    def test_over_length_fields_are_truncated_not_rejected(self):
        data = dict(GOOD, label="automation_candidate", missing_info=[],
                    rationale="x" * 5000, verification_steps=["y" * 500] + ["ok"] * 40)
        notes = clamp_classification(data)
        self.assertEqual(validate_classification(data), [],
                         "clamped data must then pass validation")
        self.assertEqual(data["label"], "automation_candidate", "the label survives")
        self.assertLessEqual(len(data["rationale"]), 2000)
        self.assertLessEqual(len(data["verification_steps"]), 20)
        self.assertTrue(all(len(s) <= 300 for s in data["verification_steps"]))
        self.assertTrue(notes, "the adjustment is reported, not silent")

    def test_clamping_never_leaves_a_magnitude_for_validation_to_reject(self):
        """clamp_classification's whole reason for existing, as a property.

        Its docstring explains the stakes: a rejected classification writes no
        entity property, so the ticket's content hash is unchanged and it is
        re-classified and re-charged on EVERY sweep, forever, and five such in
        a row abort the whole run. clamp exists so that a magnitude can never
        be the reason. Individual cases were tested; the property was not, and
        a magnitude clamp missed at one corner is a permanent recurring charge
        on that ticket rather than a visible failure.

        Every combination of the magnitude edges, which is what a fuzz over
        40,000 random inputs was really sampling.
        """
        edges = {
            "confidence": [0, 1, 0.5, -3, 95, True, False, 1.0000001, -1e-7, 1e308],
            "rationale": ["ok", "", "   ", "\n\t ", "x" * 1999, "x" * 2000, "x" * 2001,
                          "y" * 9000, " " * 3000],
            "missing_info": [[], ["a"], ["b" * 299], ["c" * 300], ["d" * 301],
                             ["e"] * 20, ["f"] * 21, ["g" * 500] * 25],
        }
        checked = 0
        for conf, rat, items in itertools.product(
                edges["confidence"], edges["rationale"], edges["missing_info"]):
            data = {"label": "needs_more_info", "rationale": rat,
                    "missing_info": list(items), "verification_steps": list(items),
                    "confidence": conf}
            clamp_classification(data)
            errors = validate_classification(data)
            self.assertEqual(errors, [], f"clamped but still rejected: conf={conf!r} "
                                         f"len(rationale)={len(rat)} items={len(items)}")
            checked += 1
        self.assertGreater(checked, 500, "the product collapsed; this proves nothing")

    def test_confidence_is_clamped_into_range(self):
        data = dict(GOOD, confidence=95)
        clamp_classification(data)
        self.assertEqual(data["confidence"], 1.0)
        data = dict(GOOD, confidence=-3)
        clamp_classification(data)
        self.assertEqual(data["confidence"], 0.0)

    def test_boolean_confidence_is_coerced(self):
        data = dict(GOOD, confidence=True)
        clamp_classification(data)
        self.assertEqual(data["confidence"], 1.0)
        self.assertEqual(validate_classification(data), [])

    def test_blank_rationale_is_substituted_not_rejected(self):
        # Schema-valid (SCHEMA cannot express minLength), so rejecting it would
        # re-charge this ticket on every sweep forever.
        for blank in ("", "   ", "\n\t ", " " * 3000):
            data = dict(GOOD, rationale=blank)
            notes = clamp_classification(data)
            self.assertEqual(validate_classification(data), [], repr(blank))
            self.assertIn("no rationale", data["rationale"])
            self.assertIn("rationale was empty", notes)

    def test_valid_data_is_left_alone(self):
        data = dict(GOOD)
        self.assertEqual(clamp_classification(data), [])
        self.assertEqual(data, GOOD)


class SinkEscapingTests(unittest.TestCase):
    """Untrusted text reaches three sinks, not one."""

    def test_invisible_and_bidi_characters_never_reach_a_comment(self):
        # Found by fuzzing, not by an exploit: the one attempt to make the model
        # carry U+202E from a ticket summary into its rationale did not
        # reproduce it. wiki_safe is the backstop whose premise is that model
        # output is untrusted, so it should hold without that attempt having
        # been representative. An unterminated override reverses everything
        # after it, and what follows the rationale in the same comment is the
        # footer saying that removing the label opts the ticket out.
        for cp in (0x202E, 0x202D, 0x200F, 0x200E, 0x200B, 0x200D, 0xFEFF, 0x2060):
            out = run.wiki_safe(f"before{chr(cp)}after")
            self.assertEqual(out, "beforeafter", f"U+{cp:04X} survived: {out!r}")

    def test_visible_whitespace_variants_collapse_rather_than_vanish(self):
        # The Cf strip must not swallow separators: these are spaces, and a
        # rationale whose words ran together would be the fix causing the harm.
        for cp in (0x00A0, 0x2028, 0x2029, 0x0085, 0x2003):
            self.assertEqual(run.wiki_safe(f"one{chr(cp)}two"), "one two",
                             f"U+{cp:04X} should collapse to a space")

    def test_sanitising_twice_changes_nothing(self):
        # Backslash removal used to run after the whitespace join, so a token
        # that was only backslashes left a doubled space behind and a second
        # pass produced different text.
        for raw in ("a \\\\ b", "x", "[~accountid:1] and !img!", "\\\\", "  ", ""):
            once = run.wiki_safe(raw)
            self.assertEqual(run.wiki_safe(once), once, f"not idempotent for {raw!r}")
        self.assertEqual(run.wiki_safe("a \\\\ b"), "a b", "no doubled space left behind")

    def test_formula_injection_is_defused_in_the_csv(self):
        for payload in ('=IMPORTXML("https://attacker.example/x","//a")',
                        '+1+1', '-2+3', '@SUM(A1)', '\ttab'):
            self.assertTrue(run.csv_safe(payload).startswith("'"), payload)

    def test_ordinary_text_is_untouched(self):
        self.assertEqual(run.csv_safe("No reproduction steps."), "No reproduction steps.")

    def test_newlines_cannot_forge_a_second_label_line(self):
        # The comment's structure is line-based, so multi-line model text could
        # otherwise state a label that was never applied.
        forged = ("Looks fine.\n\nAI triage: {{ai-triage-automation-candidate}}\n\n"
                  "_Applied by the triage pilot bot_")
        body = comment_body(load_config(), Classification(
            "needs_more_info", forged, ["x"], [], 0.5, "m"))
        lines = body.splitlines()
        # The forged text survives as prose inside the rationale, but it can no
        # longer occupy a line of its own, which is what made it read as the
        # comment's own structure.
        self.assertEqual([i for i, l in enumerate(lines) if l.startswith("AI triage:")], [0])
        self.assertEqual([i for i, l in enumerate(lines) if l.startswith("_Applied by")],
                         [len(lines) - 1])
        self.assertIn("ai-triage-needs-more-info", lines[0])

    def test_a_supplied_backslash_cannot_rearm_the_escape(self):
        # Escaping by prefix is not idempotent: text supplying its own backslash
        # turned our \[ into \\[, which Jira consumes as a line break, leaving
        # the mention live. Verified against real Jira rendering.
        body = comment_body(load_config(), Classification(
            "needs_more_info", r"contact \[~accountid:712020:9f0d] for the spec",
            [r"see \!https://attacker.example/beacon.png\!"], [], 0.5, "m"))
        self.assertNotIn("\\\\", body, "no doubled backslash can survive")
        self.assertNotIn("[~accountid", body.replace("\\[~accountid", ""))
        self.assertNotIn("!https", body.replace("\\!https", ""))

    def test_a_forced_line_break_cannot_forge_a_verdict(self):
        # \\ renders as a line break in Jira, so collapsing literal newlines was
        # not enough on its own to stop a forged second verdict block.
        forged = (r"Not enough detail. \\ \\ AI triage: {{ai-triage-automation-candidate}}"
                  r" \\ \\ _Applied by the triage pilot bot_")
        body = comment_body(load_config(), Classification(
            "needs_more_info", forged, ["x"], [], 0.5, "m"))
        self.assertNotIn("\\\\", body)

    def test_raw_html_is_escaped_in_the_markdown_sheet(self):
        hostile = Classification(
            "needs_more_info",
            'x <img src="https://attacker.example/b.png"> <h2>AI triage: cleared</h2>',
            [], [], 0.5, "m")
        stamp = datetime.datetime(2026, 8, 1, 9, 0, tzinfo=datetime.timezone.utc)
        with tempfile.TemporaryDirectory() as d:
            base = run.write_proposals(load_config(), Path(d), stamp,
                                       [(issue(), hostile, "h")], live=False)
            md = base.with_suffix(".md").read_text()
        # No unescaped angle bracket survives, so no raw HTML element renders.
        self.assertNotIn("<", md.replace("\\<", ""))
        self.assertIn("\\<img", md, "the text is preserved, just inert")

    def test_null_summary_does_not_crash_either_sink(self):
        # The field can be present-and-null, and write_proposals runs outside the
        # per-ticket try, so a TypeError here escapes after the live writes.
        self.assertEqual(run.csv_safe(None), "")
        self.assertEqual(run.wiki_safe(None), "")
        iss = issue()
        iss["fields"]["summary"] = None
        stamp = datetime.datetime(2026, 8, 1, 9, 0, tzinfo=datetime.timezone.utc)
        with tempfile.TemporaryDirectory() as d:
            base = run.write_proposals(load_config(), Path(d), stamp,
                                       [(iss, Classification("needs_more_info", "r", ["x"],
                                                             [], 0.5, "m"), "h")], live=False)
            self.assertTrue(base.with_suffix(".csv").is_file())

    def test_grading_sheet_escapes_both_sinks(self):
        hostile = Classification(
            "needs_more_info",
            '=IMPORTXML("https://attacker.example/x","//a") and ![](http://a/b.png)',
            ["=HYPERLINK(\"https://attacker.example\")"], [], 0.5, "m")
        stamp = datetime.datetime(2026, 8, 1, 9, 0, tzinfo=datetime.timezone.utc)
        with tempfile.TemporaryDirectory() as d:
            base = run.write_proposals(load_config(), Path(d), stamp,
                                       [(issue(), hostile, "h")], live=False)
            with open(base.with_suffix(".csv")) as fh:
                row = next(csv.DictReader(fh))
            md = base.with_suffix(".md").read_text()
        self.assertTrue(row["rationale"].startswith("'"), row["rationale"][:20])
        self.assertTrue(row["missing_info"].startswith("'"))
        self.assertNotIn("![](", md)

    def test_nothing_untrusted_renders_live_in_the_html_report(self):
        # The report is opened in a browser, so an unescaped <img src> in a
        # rationale is the tracking beacon fired at every reviewer - the same
        # attack wiki_safe defuses for Jira, in a different sink.
        hostile = Classification(
            "needs_more_info", '<img src=x onerror=alert(1)> and <script>bad()</script>',
            ['<iframe src="https://attacker.example"></iframe>'], [], 0.5, "m")
        stamp = datetime.datetime(2026, 8, 1, 9, 0, tzinfo=datetime.timezone.utc)
        hostile_summary = issue(summary='<img src=y> "quoted" & bare')
        with tempfile.TemporaryDirectory() as d:
            base = Path(d) / "proposals-20260801-090000"
            report = run.write_comment_report(
                load_config(), base, stamp, [(hostile_summary, hostile, "h")],
                live=False, source="file",
                excluded=[{"key": "O3-2", "summary": "<b>bold</b>",
                           "open_prs": ["https://github.com/o/r/pull/1?a=1&b=2"]}])
            page = report.read_text()
        self.assertEqual(report.suffix, ".html")
        for live_markup in ("<img", "<script", "<iframe", "<b>bold"):
            self.assertNotIn(live_markup, page.replace("&lt;", ""),
                             f"{live_markup} renders live in the report")
        # Escaped, not dropped: a reviewer must still see what the model wrote.
        self.assertIn("&lt;img src=x", page)
        self.assertIn("onerror", page)

    def test_the_report_is_self_contained(self):
        # It is mailed around and opened offline; a remote stylesheet or script
        # would also be a request to a third party on every open.
        c = Classification("needs_judgment", "Clinical call.", [], [], 0.8, "m")
        stamp = datetime.datetime(2026, 8, 1, 9, 0, tzinfo=datetime.timezone.utc)
        with tempfile.TemporaryDirectory() as d:
            page = run.write_comment_report(
                load_config(), Path(d) / "p", stamp, [(issue(), c, "h")],
                live=False, source="api", excluded=[]).read_text()
        for remote in ("<script", "src=\"http", "href=\"http://cdn", "@import"):
            self.assertNotIn(remote, page)
        # Ticket links are the one exception, and they point at Jira only.
        self.assertIn(load_config()["jira"]["base_url"], page)


class CommentReportTests(unittest.TestCase):
    """The reviewable artifact: exactly what would reach a public ticket."""

    STAMP = datetime.datetime(2026, 8, 1, 9, 30, 15, tzinfo=datetime.timezone.utc)

    def _report(self, proposals, live=False, source="api", excluded=(), out=None,
                swept=None, errors=0):
        with tempfile.TemporaryDirectory() as d:
            base = Path(out or d) / "proposals-20260801-093015"
            return run.write_comment_report(load_config(), base, self.STAMP, proposals,
                                            live, source, list(excluded),
                                            swept=swept, errors=errors).read_text()

    def test_the_body_is_the_one_jira_would_receive(self):
        # Rendered from comment_body(), not restated: a report assembled
        # independently could agree with the intent and still differ from the
        # bytes, which is the only thing a reviewer is actually signing off on.
        c = Classification("needs_more_info", "No repro steps.", ["repro steps"], [], 0.8, "m")
        page = self._report([(issue(), c, "h")])
        self.assertIn(html.escape(run.comment_body(load_config(), c)), page)

    def test_a_dry_run_says_nothing_was_written(self):
        c = Classification("needs_judgment", "Clinical call.", [], [], 0.8, "m")
        page = self._report([(issue(), c, "h")])
        self.assertIn("would write", page)
        self.assertIn("nothing written", page)

    def test_a_live_run_says_the_writes_already_happened(self):
        # The same file is produced on live runs, where it is an audit trail of
        # writes that happened - not a preview of writes that did not.
        c = Classification("needs_judgment", "Clinical call.", [], [], 0.8, "m")
        page = self._report([(issue(), c, "h")], live=True)
        self.assertIn("LIVE", page)
        self.assertNotIn("would write", page)

    def test_a_replayed_run_is_flagged_as_off_the_measured_path(self):
        c = Classification("needs_judgment", "Clinical call.", [], [], 0.8, "m")
        self.assertIn("measured path", self._report([(issue(), c, "h")], source="file"))

    def test_an_api_run_carries_no_such_warning(self):
        c = Classification("needs_judgment", "Clinical call.", [], [], 0.8, "m")
        self.assertNotIn("measured path", self._report([(issue(), c, "h")], source="api"))

    def test_excluded_tickets_are_listed_with_their_evidence(self):
        # A reviewer seeing 32 where the sweep said 34 must be told which two
        # were held back, and why.
        c = Classification("needs_judgment", "Clinical call.", [], [], 0.8, "m")
        page = self._report([(issue(), c, "h")], excluded=[
            {"key": "O3-5816", "summary": "Login crashes",
             "open_prs": ["https://github.com/openmrs/openmrs-esm-core/pull/1818"]}])
        self.assertIn("already in review", page)
        self.assertIn("O3-5816", page)
        self.assertIn("openmrs-esm-core/pull/1818", page)

    def test_a_ticket_that_gets_no_comment_says_so(self):
        # Its label is already present, so plan_label_writes suppresses the
        # comment; a reviewer counting comments would otherwise over-count.
        c = Classification("needs_more_info", "No repro.", ["steps"], [], 0.8, "m")
        label = load_config()["labels"]["needs_more_info"]
        page = self._report([(issue(labels=[label]), c, "h")])
        self.assertIn("no comment", page)
        self.assertIn("0 of 1", page)

    def test_a_label_flip_names_what_is_removed(self):
        c = Classification("needs_more_info", "No repro.", ["steps"], [], 0.8, "m")
        page = self._report([(issue(labels=[AI[0]]), c, "h")])
        self.assertIn("removed:", page)
        self.assertIn(AI[0], page)

    def test_the_share_column_survives_an_empty_proposal_list(self):
        # write_comment_report is only called when there are proposals, but a
        # division by zero here would abort a run after its writes had landed.
        self.assertIn("Summary", self._report([]))

    def test_the_lede_states_the_real_denominator(self):
        # len(proposals) is not the cohort. A live run is the record of what was
        # written to public tickets, so the report must not present "however many
        # we got through" as the scope.
        c = Classification("needs_judgment", "Clinical call.", [], [], 0.8, "m")
        page = self._report([(issue(), c, "h")], swept=32)
        self.assertIn("1 ticket(s) labelled", page)
        self.assertIn("32 in scope", page)

    def test_an_errored_run_warns_that_tickets_may_be_missing(self):
        c = Classification("needs_judgment", "Clinical call.", [], [], 0.8, "m")
        page = self._report([(issue(), c, "h")], swept=32, errors=3)
        self.assertIn("did not complete cleanly", page)
        self.assertIn("3 of 32", page)

    def test_a_steady_state_sweep_does_not_cry_wolf(self):
        # The common live run: most of the cohort is already labelled and
        # skipped, so labelled + excluded is far below swept with nothing wrong.
        # Warning on that arithmetic would fire every sweep and teach operators
        # to ignore the banner that matters.
        c = Classification("needs_judgment", "Clinical call.", [], [], 0.8, "m")
        page = self._report([(issue(), c, "h")], swept=32, errors=0)
        self.assertNotIn("did not complete cleanly", page)

    def test_a_run_produces_the_report_beside_the_grading_sheet(self):
        StubGitHub.reset()
        self.addCleanup(StubGitHub.reset)
        jira = RecordingJira({"O3-1": issue()})
        with tempfile.TemporaryDirectory() as d, \
             mock.patch.dict(os.environ, {"TRIAGE_BOT_ACCOUNT_ID": "bot"}), \
             mock.patch.object(run, "jira_from_env", lambda cfg: jira), \
             mock.patch.object(run, "Classifier", lambda *a: StubClassifier(
                 Classification("needs_judgment", "Clinical call.", [], [], 0.8, "m"))), \
             contextlib.redirect_stdout(io.StringIO()) as printed, \
             contextlib.redirect_stderr(io.StringIO()):
            run.main(["--keys", "O3-1"], out=Path(d))
            reports = sorted(Path(d).glob("proposals-*.html"))
            sheets = sorted(Path(d).glob("proposals-*.csv"))
        self.assertEqual(len(reports), 1, "no comment report was written")
        # Same stamp as the sheet, so the two are trivially correlated.
        self.assertEqual(reports[0].stem, sheets[0].stem)
        self.assertIn("Comment report", printed.getvalue())


class ClassifierTests(unittest.TestCase):
    """Response handling. The request itself needs credentials (see evals/)."""

    def _classify(self, response):
        clf = Classifier("claude-opus-5", 16000, "system prompt")
        clf.client = StubAnthropic(response)
        return clf, clf.classify("TICKET: O3-1")

    def test_valid_json_is_parsed(self):
        payload = json.dumps({"label": "needs_more_info", "rationale": "No repro.",
                              "missing_info": ["repro steps"], "verification_steps": None,
                              "confidence": "0.75"})
        _, c = self._classify(FakeResponse("end_turn", [FakeBlock(payload)]))
        self.assertEqual(c.label, "needs_more_info")
        self.assertEqual(c.missing_info, ["repro steps"])
        self.assertEqual(c.verification_steps, [])  # null coerces to empty
        self.assertEqual(c.confidence, 0.75)        # string coerces to float
        self.assertFalse(c.refused)

    def test_a_boolean_confidence_survives_normalisation_to_reach_the_bool_guard(self):
        # A composition bug, not a unit bug: ClampTests already proves
        # clamp_classification coerces a boolean and says so, and
        # ValidateClassificationTests already proves the file path refuses one.
        # Both passed while this path was broken, because classify() normalises
        # with float() first and float(True) is 1.0 - so by the time either
        # guard ran, the boolean was an ordinary maximum confidence. The comment
        # posted to the public ticket then claims the model was certain when it
        # never said so, with nothing on stderr and nothing in the errors list.
        # Only a test at the seam can see it.
        payload = json.dumps({"label": "needs_judgment", "rationale": "A clinical call.",
                              "missing_info": [], "verification_steps": [],
                              "confidence": True})
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            _, c = self._classify(FakeResponse("end_turn", [FakeBlock(payload)]))
        self.assertEqual(c.confidence, 1.0)
        self.assertIn("boolean", err.getvalue(),
                      "coercing a boolean to full confidence must not be silent")

    def test_an_unusable_classification_is_refused_on_the_api_path_too(self):
        # The last check before a label reaches a public ticket, and nothing
        # tested it: the raise could be deleted and the suite stayed green.
        # ValidateClassificationTests covers the validator as a unit and
        # FileClassifierTests covers the file path, but neither runs it through
        # classify() - so an off-enum label would have been handed back as a
        # Classification and only blown up later, in run.py's label lookup,
        # after the ticket had already been chosen for a write.
        payload = json.dumps({"label": "banana", "rationale": "r", "missing_info": [],
                              "verification_steps": [], "confidence": 0.9})
        with self.assertRaises(RuntimeError) as caught:
            self._classify(FakeResponse("end_turn", [FakeBlock(payload)]))
        self.assertIn("unusable", str(caught.exception))
        self.assertIn("label must be one of", str(caught.exception))

    def test_a_numeric_string_confidence_is_still_normalised(self):
        # The bool exclusion must not cost the normalisation it sits inside.
        payload = json.dumps({"label": "needs_judgment", "rationale": "r",
                              "missing_info": [], "verification_steps": [],
                              "confidence": "0.5"})
        _, c = self._classify(FakeResponse("end_turn", [FakeBlock(payload)]))
        self.assertEqual(c.confidence, 0.5)

    def test_refusal_is_reported_not_parsed(self):
        _, c = self._classify(FakeResponse("refusal", []))
        self.assertTrue(c.refused)
        self.assertEqual(c.model, "claude-opus-5")

    def test_truncation_raises_a_diagnosable_error(self):
        # Thinking shares max_tokens on Opus 5, so the JSON can be cut off. The
        # bare JSONDecodeError this replaces recurred on every sweep, because an
        # unchanged content hash means the ticket is retried forever.
        with self.assertRaises(RuntimeError) as caught:
            self._classify(FakeResponse("max_tokens", [FakeBlock('{"label": "needs')]))
        self.assertIn("max_tokens", str(caught.exception))

    def test_oversized_response_is_clamped_not_discarded(self):
        # Discarding it would lose the label permanently and re-charge for the
        # ticket on every sweep, so classify() must clamp rather than reject.
        payload = json.dumps({"label": "automation_candidate", "rationale": "x" * 5000,
                              "missing_info": [], "verification_steps": ["y" * 500],
                              "confidence": 95})
        with contextlib.redirect_stderr(io.StringIO()) as err:
            _, c = self._classify(FakeResponse("end_turn", [FakeBlock(payload)]))
        self.assertEqual(c.label, "automation_candidate")
        self.assertLessEqual(len(c.rationale), 2000)
        self.assertEqual(c.confidence, 1.0)
        self.assertTrue(all(len(s) <= 300 for s in c.verification_steps))
        self.assertIn("adjusted", err.getvalue())

    def test_missing_text_block_raises(self):
        with self.assertRaises(RuntimeError):
            self._classify(FakeResponse("end_turn", [FakeBlock("", type="thinking")]))

    def test_request_carries_the_schema_and_refusal_fallback(self):
        payload = json.dumps({"label": "needs_judgment", "rationale": "r",
                              "missing_info": [], "verification_steps": [],
                              "confidence": 0.5})
        clf, _ = self._classify(FakeResponse("end_turn", [FakeBlock(payload)]))
        sent = clf.client.kwargs
        self.assertEqual(sent["model"], "claude-opus-5")
        self.assertEqual(sent["max_tokens"], 16000)
        # The "default" fallback form pairs with the -07-01 beta; the array form
        # uses -06-01, and crossing them is rejected.
        self.assertIn("server-side-fallback-2026-07-01", sent["betas"])
        self.assertEqual(sent["extra_body"]["fallbacks"], "default")
        fmt = sent["extra_body"]["output_config"]["format"]
        self.assertEqual(fmt["type"], "json_schema")
        self.assertEqual(fmt["schema"]["properties"]["label"]["enum"], LABEL_KEYS)


GOOD = {"label": "needs_more_info", "rationale": "No repro.",
        "missing_info": ["repro steps"], "verification_steps": [], "confidence": 0.9}


class DecisionChainInvariantTests(unittest.TestCase):
    """Drive main() over the decision chain's whole state space.

    Every regression this pipeline has shipped lived in a seam between two
    individually-correct rules - the label/comment ordering, the content+prompt
    idempotency key, the source term, the error breaker. Reviewers have to
    imagine those seams one at a time. This enumerates them instead, and asserts
    the handful of properties that must hold in EVERY state rather than checking
    named scenarios.
    """

    LABELS = ([], [AI[0]], [AI[2]])
    PROPERTIES = ("absent", "matching", "stale-hash", "stale-prompt", "replayed")
    STATUSES = ("To Do", "In Progress")

    # Every classification carries a hostile payload, so invariant 4 is checked
    # against real constructs in every state rather than against inert prose.
    # Without this the harness passed with all escaping removed.
    HOSTILE_RATIONALE = (r"contact \[~accountid:712020:9f0d] and see "
                         r"!https://attacker.example/b.png! <img src=x> \\ \\ done")
    HOSTILE_ITEM = r"ping \[~accountid:712020:def] <b>now</b>"

    def setUp(self):
        # Every case sets its own open-PR answer; starting from a clean slate
        # keeps a leaked one from silently turning a case into a skip.
        StubGitHub.reset()
        self.addCleanup(StubGitHub.reset)

    def _issue(self, labels, status, opted_out):
        histories = [label_change("u1", AI[2], "")] if opted_out else []
        iss = issue(labels=list(labels), histories=histories, status={"name": status})
        iss["changelog"]["total"] = len(histories)
        return iss

    def _property_for(self, kind, chash, label):
        if kind == "absent":
            return None
        base = {"contentHash": chash, "label": label, "prompt": PROMPT_VERSION,
                "source": "api", "classifier": "m"}
        if kind == "stale-hash":
            base["contentHash"] = "0" * 16
        elif kind == "stale-prompt":
            base["prompt"] = "v0"
        elif kind == "replayed":
            base["source"] = "file"
        return base

    def _run_once(self, jira, out, decided, live, force, path=None):
        argv = ["--keys", "O3-1"]
        if live:
            argv.append("--live")
        if force:
            argv.append("--force")
        if path:
            argv += ["--classifications", str(path)]
        with mock.patch.dict(os.environ, {"TRIAGE_BOT_ACCOUNT_ID": "bot"}), \
             mock.patch.object(run, "jira_from_env", lambda cfg: jira), \
             mock.patch.object(run, "Classifier",
                               lambda *a: StubClassifier(decided)), \
             contextlib.redirect_stdout(io.StringIO()), \
             contextlib.redirect_stderr(io.StringIO()):
            return run.main(argv, out=out)

    def test_invariants_hold_across_the_state_space(self):
        cfg = load_config()
        scope_status = cfg["jira"]["scope_status"]
        checked = 0
        self.addCleanup(StubGitHub.reset)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            for (labels, prop_kind, status, opted_out, force, decided_label,
                 open_pr) in itertools.product(
                    self.LABELS, self.PROPERTIES, self.STATUSES, (False, True),
                    (False, True), ("needs_more_info", "automation_candidate"),
                    (False, True)):
                decided = Classification(decided_label, self.HOSTILE_RATIONALE,
                                         [self.HOSTILE_ITEM], [], 0.9, "m")
                label_name = cfg["labels"][decided_label]
                StubGitHub.reset({"O3-1": ["https://github.com/openmrs/r/pull/1"]}
                                 if open_pr else {})
                jira = RecordingJira({"O3-1": self._issue(labels, status, opted_out)})
                text = ctx.assemble(StubJira(), jira.issues["O3-1"], None, ["bot"])
                stored = self._property_for(prop_kind, ctx.content_hash(text), label_name)
                if stored:
                    jira.properties["O3-1"] = stored
                rc = self._run_once(jira, out, decided, live=True, force=force)
                writes = jira.writes
                state = (f"labels={labels} prop={prop_kind} status={status} "
                         f"opted_out={opted_out} force={force} decided={decided_label} "
                         f"open_pr={open_pr}")
                checked += 1

                # 1. An opt-out is permanent and unconditional.
                if opted_out:
                    self.assertEqual(writes, [], f"wrote to an opted-out ticket: {state}")
                    continue
                # 2. A ticket outside scope is never written to in live mode.
                if status != scope_status:
                    self.assertEqual(writes, [], f"wrote out of scope: {state}")
                    continue
                # 2b. Nor is one that is already in review. Labelling it invites
                #     the removal the pilot counts as a permanent opt-out.
                if open_pr:
                    self.assertEqual(writes, [], f"wrote to a ticket with an open PR: {state}")
                    continue
                # 3. At most one comment per run, and a comment only ever
                #    accompanies a label that is new to the ticket.
                comments = [w for w in writes if w[0] == "comment"]
                self.assertLessEqual(len(comments), 1, f"multiple comments: {state}")
                if comments:
                    added = [w for w in writes if w[0] == "labels" and label_name in w[2]]
                    self.assertTrue(added, f"commented without adding the label: {state}")
                # 4. Nothing untrusted renders live in a comment body.
                for comment in comments:
                    body = comment[2]
                    self.assertNotIn("\\\\", body, f"doubled backslash: {state}")
                    for char in ("[", "!", "<"):
                        self.assertNotIn(char, body.replace("\\" + char, "")
                                         .replace("{{", "").replace("}}", ""),
                                         f"unescaped {char!r}: {state}")
                # 5. Exactly one ai-triage label afterwards, and it is the
                #    decided one, whenever any write happened.
                if writes:
                    present = [l for l in jira.issues["O3-1"]["fields"]["labels"]
                               if l in AI]
                    self.assertEqual(present, [label_name], f"label set wrong: {state}")
                # 6. Success is reported iff nothing errored.
                self.assertEqual(rc, 0, f"nonzero exit with no fault: {state}")
                # 7. Liveness. The invariants above are all "nothing bad
                #    happens"; on their own they are satisfied by a pipeline that
                #    does nothing at all, and dropping a term from the
                #    idempotency key is precisely a failure to act. So the skip
                #    decision is specified in full: a ticket is skipped exactly
                #    when it already carries a label whose stored triage still
                #    matches on all three of content, prompt and classifier.
                triaged = (stored is not None
                           and stored["contentHash"] == ctx.content_hash(text)
                           and stored["prompt"] == cfg["prompt"]["version"]
                           and stored["source"] == "api")
                if labels and triaged and not force:
                    self.assertEqual(writes, [], f"should have skipped: {state}")
                else:
                    self.assertTrue(any(w[0] == "property" for w in writes),
                                    f"should have re-classified: {state}")

                # 8. Idempotence: an immediate identical re-run must not
                #    comment again. This is the property the flip-flop broke.
                jira.writes.clear()
                self._run_once(jira, out, decided, live=True, force=force)
                self.assertEqual([w for w in jira.writes if w[0] == "comment"], [],
                                 f"re-commented on an unchanged ticket: {state}")
        self.assertGreater(checked, 200, "the space should be non-trivial")

    def test_alternating_the_two_paths_converges(self):
        """Alternating the API sweep and a replay run must stop writing.

        One supersession is legitimate: the pinned-model path is authoritative,
        so replacing a replayed label - and commenting to explain the new one -
        is correct. The defect the flip-flop introduced was that it never
        *terminated*: each sweep swapped the label back and comment again, every
        four hours, forever. So the property is convergence, not silence.
        """
        # The two paths must DISAGREE on the label here: agreement hides the bug,
        # because plan_label_writes only comments on a label new to the ticket.
        # A replay classifier and the pinned model will not always agree.
        for first_is_api, (api_label, file_label) in itertools.product(
                (True, False),
                (("needs_more_info", "needs_judgment"),
                 ("automation_candidate", "needs_more_info"))):
            decided = Classification(api_label, "Because.", ["x"], [], 0.9, "m")
            jira = RecordingJira({"O3-1": issue()})
            with tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp)
                text = ctx.assemble(StubJira(), jira.issues["O3-1"], None, ["bot"])
                path = out / "c.json"
                path.write_text(json.dumps({
                    "prompt_version": PROMPT_VERSION, "classifier": "agent",
                    "classifications": {"O3-1": {
                        "content_hash": ctx.content_hash(text), "label": file_label,
                        "rationale": "Because.", "missing_info": [],
                        "verification_steps": [], "confidence": 0.9}},
                }))
                order = [None, path] if first_is_api else [path, None]
                comments = []
                for round_index in range(6):
                    jira.writes.clear()
                    self._run_once(jira, out, decided, live=True, force=False,
                                   path=order[round_index % 2])
                    comments.append(len([w for w in jira.writes if w[0] == "comment"]))
                state = (f"api first: {first_is_api}, api={api_label}, "
                         f"file={file_label}, comments per round: {comments}")
                # Settles within the first two rounds: the initial label, then at
                # most one supersession. Nothing after that.
                self.assertEqual(comments[2:], [0, 0, 0, 0], f"did not converge - {state}")
                self.assertLessEqual(sum(comments), 2, f"too many comments - {state}")


class ManifestRoundTripTests(unittest.TestCase):
    """Gather -> classify from the manifest -> apply, with nothing hand-copied."""

    def test_a_file_built_only_from_the_manifest_applies_cleanly(self):
        jira = RecordingJira({"O3-1": issue(), "O3-2": issue()})
        jira.issues["O3-2"]["key"] = "O3-2"
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            with mock.patch.dict(os.environ, {"TRIAGE_BOT_ACCOUNT_ID": "bot"}), \
                 mock.patch.object(run, "jira_from_env", lambda cfg: jira), \
                 mock.patch.object(run, "Classifier", lambda *a: None), \
                 contextlib.redirect_stdout(io.StringIO()), \
                 contextlib.redirect_stderr(io.StringIO()):
                run.main(["--no-classify", "--keys", "O3-1,O3-2"], out=out)
                manifest = json.loads((out / "manifest.json").read_text())

                # Build entries using ONLY what the manifest advertises.
                entry_schema = manifest["entry_schema"]
                self.assertIn("content_hash", entry_schema["required"],
                              "a consumer obeying this schema must be able to include it")
                self.assertIn("content_hash", entry_schema["properties"])
                entries = {}
                for key, info in manifest["tickets"].items():
                    self.assertTrue((out / info["context"]).is_file())
                    entries[key] = {
                        "content_hash": info["content_hash"],
                        "label": "needs_more_info", "rationale": "No repro.",
                        "missing_info": ["repro steps"], "verification_steps": [],
                        "confidence": 0.9,
                    }
                    self.assertEqual(
                        sorted(entries[key]), sorted(entry_schema["required"]),
                        "the schema's required set must be exactly what an entry needs")
                path = out / "c.json"
                path.write_text(json.dumps({
                    "prompt_version": manifest["prompt_version"],
                    "classifier": "round-trip", "classifications": entries,
                }))
                rc = run.main(["--live", "--keys", "O3-1,O3-2",
                               "--classifications", str(path)], out=out)
        self.assertEqual(rc, 0)
        self.assertEqual(sorted(w[1] for w in jira.writes if w[0] == "labels"),
                         ["O3-1", "O3-2"])

    def test_the_apply_run_does_not_overwrite_the_gather_manifest(self):
        jira = RecordingJira({"O3-1": issue(), "O3-2": issue()})
        jira.issues["O3-2"]["key"] = "O3-2"
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            with mock.patch.dict(os.environ, {"TRIAGE_BOT_ACCOUNT_ID": "bot"}), \
                 mock.patch.object(run, "jira_from_env", lambda cfg: jira), \
                 mock.patch.object(run, "Classifier", lambda *a: None), \
                 contextlib.redirect_stdout(io.StringIO()), \
                 contextlib.redirect_stderr(io.StringIO()):
                run.main(["--no-classify", "--keys", "O3-1,O3-2"], out=out)
                before = (out / "manifest.json").read_text()
                text = ctx.assemble(StubJira(), jira.issues["O3-1"], None, ["bot"])
                path = out / "c.json"
                path.write_text(json.dumps({
                    "prompt_version": PROMPT_VERSION, "classifier": "agent",
                    "classifications": {"O3-1": dict(GOOD, content_hash=ctx.content_hash(text))},
                }))
                run.main(["--live", "--keys", "O3-1", "--classifications", str(path)], out=out)
                after = (out / "manifest.json").read_text()
        self.assertEqual(before, after,
                         "a narrower apply run must not clobber the gather's manifest")

    def _gather(self, jira, keys, out):
        with mock.patch.dict(os.environ, {"TRIAGE_BOT_ACCOUNT_ID": "bot"}), \
             mock.patch.object(run, "jira_from_env", lambda cfg: jira), \
             mock.patch.object(run, "Classifier", lambda *a: None), \
             contextlib.redirect_stdout(io.StringIO()), \
             contextlib.redirect_stderr(io.StringIO()):
            run.main(["--no-classify", "--keys", keys], out=out)
        return json.loads((out / "manifest.json").read_text())

    def test_already_triaged_tickets_are_still_offered(self):
        # Not a permanent skip: a live run re-classifies these after a prompt bump
        # or an edit, and the gather step is a dry run where `unchanged` is
        # unconditionally True - so gating the manifest on it dropped the entire
        # re-triage backlog.
        jira = RecordingJira({"O3-1": issue(labels=[AI[2]])})
        with tempfile.TemporaryDirectory() as d:
            manifest = self._gather(jira, "O3-1", Path(d))
        self.assertEqual(sorted(manifest["tickets"]), ["O3-1"])

    def test_complete_is_false_when_any_ticket_failed(self):
        # An errored ticket is simply absent from tickets[], so a consumer
        # trusting `complete` would never classify it.
        class Flaky(RecordingJira):
            def issue(self, key, fields, expand_changelog=False):
                if key == "O3-2":
                    raise JiraError("500 Internal Server Error")
                return self.issues[key]

        jira = Flaky({"O3-1": issue(), "O3-2": issue()})
        jira.issues["O3-2"]["key"] = "O3-2"
        with tempfile.TemporaryDirectory() as d:
            manifest = self._gather(jira, "O3-1,O3-2", Path(d))
        self.assertEqual(sorted(manifest["tickets"]), ["O3-1"])
        self.assertFalse(manifest["complete"])

    def test_skipped_tickets_are_not_offered_for_classification(self):
        # An opted-out ticket must not be advertised: the pilot promised never to
        # touch it again, so classifying it is wasted effort on a dead ticket.
        jira = RecordingJira({
            "O3-1": issue(),
            "O3-2": issue(histories=[label_change("u1", AI[2], "")]),
        })
        jira.issues["O3-2"]["key"] = "O3-2"
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            with mock.patch.dict(os.environ, {"TRIAGE_BOT_ACCOUNT_ID": "bot"}), \
                 mock.patch.object(run, "jira_from_env", lambda cfg: jira), \
                 mock.patch.object(run, "Classifier", lambda *a: None), \
                 contextlib.redirect_stdout(io.StringIO()), \
                 contextlib.redirect_stderr(io.StringIO()):
                run.main(["--no-classify", "--keys", "O3-1,O3-2"], out=out)
            manifest = json.loads((out / "manifest.json").read_text())
        self.assertEqual(sorted(manifest["tickets"]), ["O3-1"])


class ValidateClassificationTests(unittest.TestCase):
    def test_valid_object_has_no_errors(self):
        self.assertEqual(validate_classification(dict(GOOD)), [])

    def test_missing_required_field_is_reported(self):
        data = dict(GOOD)
        del data["confidence"]
        self.assertIn("missing 'confidence'", validate_classification(data))

    def test_label_outside_the_enum_is_reported(self):
        self.assertTrue(any("must be one of" in e for e in
                            validate_classification(dict(GOOD, label="looks_fine"))))

    def test_wrong_types_are_reported(self):
        self.assertTrue(any("array of strings" in e for e in
                            validate_classification(dict(GOOD, missing_info="repro"))))
        self.assertTrue(any("must be a number" in e for e in
                            validate_classification(dict(GOOD, confidence="high"))))

    def test_unexpected_field_is_reported(self):
        self.assertTrue(any("unexpected field" in e for e in
                            validate_classification(dict(GOOD, sneaky="x"))))

    def test_confidence_outside_zero_to_one_is_reported(self):
        # The prompt asks for an honest 0-1 estimate; nothing in the schema can
        # enforce a range, and it is rendered into the comment and the CSV.
        for bad in (1.5, -0.2, 99):
            self.assertTrue(any("between 0 and 1" in e for e in
                                validate_classification(dict(GOOD, confidence=bad))), bad)
        for ok in (0, 1, 0.5):
            self.assertEqual(validate_classification(dict(GOOD, confidence=ok)), [], ok)

    def test_oversized_text_is_reported(self):
        # A file is not bound by the prompt's two-sentence rule, and this text is
        # posted to a public ticket.
        long_rationale = "x" * 5000
        self.assertTrue(any("over the" in e for e in
                            validate_classification(dict(GOOD, rationale=long_rationale))))
        self.assertTrue(any("over the" in e for e in
                            validate_classification(dict(GOOD, missing_info=["y" * 500]))))
        self.assertTrue(any("over the" in e for e in
                            validate_classification(dict(GOOD, missing_info=["z"] * 50))))


class FileClassifierTests(unittest.TestCase):
    """Replaying classifications made outside the pipeline."""

    def _write(self, tmp, entry, prompt_version=None, classifier="agent"):
        prompt_version = PROMPT_VERSION if prompt_version is None else prompt_version
        path = Path(tmp) / "c.json"
        path.write_text(json.dumps({
            "prompt_version": prompt_version, "classifier": classifier,
            "classifications": {"O3-1": entry},
        }))
        return path

    def test_replays_the_classification_for_a_matching_context(self):
        text = "TICKET: O3-1\n"
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(tmp, dict(GOOD, content_hash=ctx.content_hash(text)))
            fc = run.FileClassifier(path, PROMPT_VERSION)
        c = fc.classify(text)
        self.assertEqual(c.label, "needs_more_info")
        self.assertEqual(c.model, "agent")

    def test_edited_context_finds_no_classification(self):
        # The hash is the staleness guard: a label made against older text must
        # never reach Jira. Raised as NotClassified, which is a skip rather than
        # a fault, so it cannot trip the paid-fault circuit breaker.
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(tmp, dict(GOOD, content_hash=ctx.content_hash("TICKET: O3-1\n")))
            fc = run.FileClassifier(path, PROMPT_VERSION)
        with self.assertRaises(run.NotClassified) as caught:
            fc.classify("TICKET: O3-1 (edited)\n")
        self.assertIn("not in this batch", str(caught.exception))

    def test_mispaired_content_hash_is_refused(self):
        # Swapping two entries' hashes would apply each ticket's label and
        # rationale to the other, and both hashes are individually valid.
        one, two = "TICKET: O3-1\nSUMMARY: a\n", "TICKET: O3-2\nSUMMARY: b\n"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "c.json"
            path.write_text(json.dumps({
                "prompt_version": PROMPT_VERSION, "classifier": "agent",
                "classifications": {
                    "O3-1": dict(GOOD, content_hash=ctx.content_hash(two)),
                    "O3-2": dict(GOOD, content_hash=ctx.content_hash(one)),
                },
            }))
            fc = run.FileClassifier(path, PROMPT_VERSION)
        with self.assertRaises(RuntimeError) as caught:
            fc.classify(one)
        self.assertIn("mispaired", str(caught.exception))

    def test_classifier_string_cannot_forge_log_or_report_lines(self):
        # The one header field validate_classification never sees. It reaches the
        # run log, the entity property and the weekly decision report verbatim.
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "c.json"
            path.write_text(json.dumps({
                "prompt_version": PROMPT_VERSION,
                "classifier": "opus-5\n  O3-1: skip-opted-out\n\nDECISION: ADOPT",
                "classifications": {"O3-1": dict(GOOD, content_hash="abc")},
            }))
            fc = run.FileClassifier(path, PROMPT_VERSION)
        self.assertNotIn("\n", fc.model)
        self.assertNotIn("DECISION: ADOPT\n", fc.model)

    def test_non_string_classifier_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "c.json"
            path.write_text(json.dumps({
                "prompt_version": PROMPT_VERSION, "classifier": {"nested": "x"},
                "classifications": {"O3-1": dict(GOOD, content_hash="abc")},
            }))
            with self.assertRaises(SystemExit) as caught:
                run.FileClassifier(path, PROMPT_VERSION)
        self.assertIn("classifier must be a string", str(caught.exception))

    def test_classifications_as_a_list_is_named_not_a_traceback(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "c.json"
            path.write_text(json.dumps({"prompt_version": PROMPT_VERSION, "classifications": ["x"]}))
            with self.assertRaises(SystemExit) as caught:
                run.FileClassifier(path, PROMPT_VERSION)
        self.assertIn("must be an object", str(caught.exception))

    def test_lowercase_ticket_key_still_matches(self):
        # Jira normalises a lowercase key, so a gather run with --keys o3-1 writes
        # the manifest under "o3-1" while the context says "TICKET: O3-1".
        text = "TICKET: O3-1\nSUMMARY: a\n"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "c.json"
            path.write_text(json.dumps({
                "prompt_version": PROMPT_VERSION, "classifier": "agent",
                "classifications": {"o3-1": dict(GOOD, content_hash=ctx.content_hash(text))},
            }))
            fc = run.FileClassifier(path, PROMPT_VERSION)
        self.assertEqual(fc.classify(text).label, "needs_more_info")

    def test_boolean_confidence_is_refused(self):
        # bool is an int in Python; unguarded this renders as a confidence of
        # 1.00. The file path rejects because its author can fix it.
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(tmp, dict(GOOD, content_hash="abc", confidence=True))
            with self.assertRaises(SystemExit) as caught:
                run.FileClassifier(path, PROMPT_VERSION)
        self.assertIn("must be a number", str(caught.exception))

    def test_non_string_content_hash_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(tmp, dict(GOOD, content_hash=12345))
            with self.assertRaises(SystemExit) as caught:
                run.FileClassifier(path, PROMPT_VERSION)
        self.assertIn("must be a string", str(caught.exception))

    def test_prompt_version_mismatch_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(tmp, dict(GOOD, content_hash="abc"), prompt_version="v0")
            with self.assertRaises(SystemExit) as caught:
                run.FileClassifier(path, PROMPT_VERSION)
        self.assertIn("v0", str(caught.exception))

    def test_missing_content_hash_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(tmp, dict(GOOD))
            with self.assertRaises(SystemExit) as caught:
                run.FileClassifier(path, PROMPT_VERSION)
        self.assertIn("content_hash", str(caught.exception))

    def test_invalid_classification_is_refused_before_any_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(tmp, dict(GOOD, content_hash="abc", label="not_a_label"))
            with self.assertRaises(SystemExit) as caught:
                run.FileClassifier(path, PROMPT_VERSION)
        self.assertIn("not a valid classification", str(caught.exception))

    def test_missing_file_is_named(self):
        with self.assertRaises(SystemExit) as caught:
            run.FileClassifier(Path("/nonexistent/classifications.json"), "v1")
        self.assertIn("no such file", str(caught.exception))

    def test_malformed_json_is_named(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "c.json"
            path.write_text("{not json")
            with self.assertRaises(SystemExit) as caught:
                run.FileClassifier(path, PROMPT_VERSION)
        self.assertIn("not valid JSON", str(caught.exception))

    def test_non_object_document_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "c.json"
            path.write_text('["nope"]')
            with self.assertRaises(SystemExit) as caught:
                run.FileClassifier(path, PROMPT_VERSION)
        self.assertIn("expected a JSON object", str(caught.exception))

    def test_empty_classifications_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "c.json"
            path.write_text(json.dumps({"prompt_version": PROMPT_VERSION, "classifications": {}}))
            with self.assertRaises(SystemExit) as caught:
                run.FileClassifier(path, PROMPT_VERSION)
        self.assertIn("no classifications", str(caught.exception))

    def test_local_faults_fail_before_any_jira_call(self):
        # A bad file is a purely local fault; failing after a full JQL sweep
        # wastes the sweep and buries the error under later stdout.
        def explode(cfg):
            raise AssertionError("Jira must not be contacted for a local fault")

        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "c.json"
            bad.write_text(json.dumps({
                "prompt_version": PROMPT_VERSION,
                "classifications": {"O3-1": dict(GOOD, content_hash="a", confidence=42)},
            }))
            with mock.patch.object(run, "jira_from_env", explode), \
                 contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    run.main(["--keys", "O3-1", "--classifications", str(bad)], out=Path(tmp))
                with self.assertRaises(SystemExit):
                    run.main(["--keys", "O3-1", "--no-classify",
                              "--classifications", str(bad)], out=Path(tmp))

    def test_duplicate_content_hash_is_refused(self):
        # Lookup is by hash, so a repeat would silently overwrite and which
        # label landed would depend on dict order.
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "c.json"
            path.write_text(json.dumps({
                "prompt_version": PROMPT_VERSION, "classifier": "agent",
                "classifications": {
                    "O3-1": dict(GOOD, content_hash="same"),
                    "O3-2": dict(GOOD, content_hash="same", label="needs_judgment",
                                 missing_info=[]),
                },
            }))
            with self.assertRaises(SystemExit) as caught:
                run.FileClassifier(path, PROMPT_VERSION)
        self.assertIn("share content_hash", str(caught.exception))


class ConfigTests(unittest.TestCase):
    def test_config_labels_match_classifier_keys(self):
        cfg = load_config()
        self.assertEqual(set(cfg["labels"].keys()), set(LABEL_KEYS))
        self.assertEqual(SCHEMA["properties"]["label"]["enum"], LABEL_KEYS)

    def test_labels_avoid_jira_restricted_characters(self):
        # Jira Cloud rejects spaces and /:;,.?&[]()#^*@! in labels.
        cfg = load_config()
        for value in cfg["labels"].values():
            self.assertNotRegex(value, r"[\s/:;,.?&\[\]()#^*@!]")

    def test_scope_jql_formats_with_pinned_date(self):
        cfg = load_config()
        jql = cfg["jira"]["scope_jql"].format(since=cfg["jira"]["cohort_created_since"])
        self.assertIn('created >= "20', jql)
        self.assertNotIn("{since}", jql)

    def test_dev_panel_clause_is_a_substring_of_scope_jql(self):
        # run.py and preflight.py both strip this clause by substring match to
        # retry the sweep; drift would silently disable both fallbacks.
        cfg = load_config()
        self.assertIn(cfg["jira"]["dev_panel_clause"], cfg["jira"]["scope_jql"])

    def test_cohort_jql_is_wider_than_scope_jql(self):
        # The metrics denominator must still find tickets the bot labelled that
        # have since left "To Do" or gained a linked PR.
        cfg = load_config()
        cohort = cfg["jira"]["cohort_jql"].format(since=cfg["jira"]["cohort_created_since"])
        self.assertNotIn("{since}", cohort)
        self.assertNotIn(cfg["jira"]["scope_status"], cohort)
        self.assertNotIn(cfg["jira"]["dev_panel_clause"], cohort)

    def test_scope_status_matches_the_status_in_scope_jql(self):
        # These are two statements of the same fact; drift between them would
        # make every swept ticket look out of scope and silently stop all writes.
        cfg = load_config()
        self.assertIn(f'status = "{cfg["jira"]["scope_status"]}"', cfg["jira"]["scope_jql"])


if __name__ == "__main__":
    unittest.main()
