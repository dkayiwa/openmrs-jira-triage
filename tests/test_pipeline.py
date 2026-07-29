"""Offline tests for the pure pipeline logic (no network, no API key).

Fixture shapes mirror real Jira Cloud v2 responses: changelog label lists,
paginated changelog envelopes and missing-property status codes were all
verified live against openmrs.atlassian.net. Classifier response handling is
covered here against a stub client; only the real HTTP request needs
credentials, and that is exercised by evals/run_evals.py.
"""
from __future__ import annotations

import contextlib
import csv
import datetime
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from triage import context as ctx  # noqa: E402
from triage import metrics, run  # noqa: E402
from triage.classifier import LABEL_KEYS, SCHEMA, Classification, Classifier  # noqa: E402
from triage.jira import JiraClient, JiraError  # noqa: E402
from triage.metrics import decide, sla_met  # noqa: E402
from triage.run import (  # noqa: E402
    _load_dotenv,
    bot_identity_error,
    comment_body,
    load_config,
    plan_label_writes,
    plan_ticket,
)
from triage.state import TicketState, inspect  # noqa: E402

AI = ["ai-triage-automation-candidate", "ai-triage-needs-judgment", "ai-triage-needs-more-info"]


def issue(labels=(), histories=(), comments=(), **fields):
    base = {
        "summary": "Fix the widget", "description": "It is broken.",
        "labels": list(labels), "issuelinks": [], "parent": None,
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
        self.calls.append({"url": url, **(params or {})})
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

    def add_comment(self, key, body):
        self.writes.append(("comment", key, body))


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

    def test_acceptance_criteria_included_when_configured(self):
        text = ctx.assemble(StubJira(), issue(customfield_1="Given X then Y"), "customfield_1", [])
        self.assertIn("ACCEPTANCE CRITERIA:\nGiven X then Y", text)

    def test_linked_tickets_listed(self):
        links = [{"type": {"outward": "blocks", "inward": "is blocked by"},
                  "outwardIssue": {"key": "O3-9", "fields": {"summary": "Other"}}}]
        text = ctx.assemble(StubJira(), issue(issuelinks=links), None, [])
        self.assertIn("- blocks O3-9: Other", text)

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


class WritePlanTests(unittest.TestCase):
    def test_fresh_ticket_gets_label_and_comment(self):
        self.assertEqual(plan_label_writes([], AI[0]), ([AI[0]], [], True))

    def test_same_label_present_is_quiet(self):
        self.assertEqual(plan_label_writes([AI[0]], AI[0]), ([], [], False))

    def test_label_flip_removes_stale_and_comments(self):
        self.assertEqual(plan_label_writes([AI[2]], AI[0]), ([AI[0]], [AI[2]], True))


def stub_client(pages):
    client = JiraClient("https://example.invalid")
    client.session = StubSession(pages)
    return client


class PaginationTests(unittest.TestCase):
    def test_comments_page_past_truncated_embedded_list(self):
        client = stub_client([StubResponse({"comments": [{"body": "third"}], "total": 3})])
        embedded = {"comments": [{"body": "first"}, {"body": "second"}], "total": 3}
        result = client.comments("O3-1", embedded)
        self.assertEqual([c["body"] for c in result], ["first", "second", "third"])
        self.assertEqual(client.session.calls[0]["startAt"], 2)

    def test_changelog_pages_past_embedded_100_entry_limit(self):
        # expand=changelog truncates at 100 and drops the *newest* entries -
        # exactly where an opt-out removal lives. The dedicated endpoint wraps
        # items in "values", not "histories".
        embedded = {"histories": [{"id": str(i)} for i in range(100)], "total": 102}
        client = stub_client([StubResponse({"values": [{"id": "100"}, {"id": "101"}], "total": 102})])
        result = client.changelog("O3-1", embedded)
        self.assertEqual(len(result), 102)
        self.assertEqual(result[-1]["id"], "101")
        self.assertIn("/changelog", client.session.calls[0]["url"])
        self.assertEqual(client.session.calls[0]["startAt"], 100)

    def test_untruncated_changelog_costs_no_extra_request(self):
        client = stub_client([])
        result = client.changelog("O3-1", {"histories": [{"id": "1"}], "total": 1})
        self.assertEqual(len(result), 1)
        self.assertEqual(client.session.calls, [])

    def test_opt_out_in_the_paged_tail_is_detected(self):
        # The whole point of paging: a removal past entry 100 must still count.
        embedded = {"histories": [{"items": []}] * 100, "total": 101}
        client = stub_client([StubResponse(
            {"values": [label_change("u1", AI[2], "")], "total": 101})])
        st = inspect(issue(), AI, "bot", client.changelog("O3-1", embedded))
        self.assertTrue(st.opted_out)


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

    def test_unset_bot_id_is_not_a_mismatch(self):
        self.assertIsNone(bot_identity_error(self._Jira("bot"), None))


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


class EvalImportTests(unittest.TestCase):
    """A grade must stay pinned to the context it was actually made against."""

    HEADER = ["key", "url", "summary", "proposed_label", "confidence", "rationale",
              "missing_info", "verification_steps", "content_hash",
              "grade(ok/wrong)", "correct_label", "grader_notes"]

    def _import(self, context_text, graded_hash, label="needs_more_info"):
        module = load_evals_module()
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / "contexts").mkdir()
            (d / "contexts" / "O3-1.txt").write_text(context_text)
            proposals = d / "proposals.csv"
            with open(proposals, "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(self.HEADER)
                w.writerow(["O3-1", "u", "s", label, "0.90", "r", "", "",
                            graded_hash, "ok", "", ""])
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

    def test_context_overwritten_since_grading_is_skipped(self):
        rows, log = self._import("TICKET: O3-1 (edited)\n",
                                 ctx.content_hash("TICKET: O3-1\n"))
        self.assertEqual(rows, [])
        self.assertIn("changed since grading", log)

    def test_sheet_without_a_hash_column_still_imports(self):
        # Proposals CSVs produced before the column existed stay usable.
        text = "TICKET: O3-1\n"
        rows, _ = self._import(text, "")
        self.assertEqual([r["key"] for r in rows], ["O3-1"])


class MetricsJira:
    def __init__(self, issues, intro_keys):
        self.issues = issues
        self.intro_keys = list(intro_keys)

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

    def _report(self, issue_json, intro_keys=("O3-1",) * 5):
        cfg = load_config()
        cfg["metrics"] = dict(cfg["metrics"], pilot_launch="2026-08-01")
        jira = MetricsJira({"O3-1": issue_json}, intro_keys)
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
        self.assertIn("sorted within 24h  : 100%", report)
        self.assertIn("DECISION: ADOPT", report)

    def test_late_labeling_is_not_silently_credited(self):
        report = self._report(self._labeled("2026-08-10T09:00:00.000+0000",
                                            "2026-08-12T09:00:00.000+0000"))
        self.assertIn("sorted within 24h  : 0%", report)
        self.assertNotIn("ADOPT", report)

    def test_human_relabel_counts_as_removal_and_violation(self):
        report = self._report(issue(
            labels=[AI[0]], created="2026-08-10T09:00:00.000+0000",
            histories=[
                label_change("bot", "", AI[2], created="2026-08-10T10:00:00.000+0000"),
                label_change("u1", AI[2], AI[0], display="Maintainer",
                             created="2026-08-11T09:00:00.000+0000"),
            ]))
        self.assertIn("label removal rate : 1.00", report)
        self.assertIn("convention adds    : 1", report)
        self.assertIn("Maintainer", report)
        self.assertIn("DECISION: STOP", report)


class LiveRunTests(unittest.TestCase):
    """main() in --live: exactly one label and one comment per ticket, ever."""

    def _run(self, jira, label="needs_more_info", extra_args=()):
        classification = Classification(label, "Because.", ["repro steps"], [], 0.9, "m")
        argv = ["--live", "--keys", "O3-1", *extra_args]
        with tempfile.TemporaryDirectory() as d:
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
        self.assertEqual(jira.properties["O3-1"]["prompt"], "v1")
        jira.properties["O3-1"]["prompt"] = "v0"
        jira.writes.clear()
        row = self._run(jira)
        self.assertEqual(row["action"], "refreshed")
        self.assertEqual([w[0] for w in jira.writes], ["property"])

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


if __name__ == "__main__":
    unittest.main()
