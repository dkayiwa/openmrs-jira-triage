"""Offline tests for the pure pipeline logic (no network, no API key).

Fixture shapes mirror real Jira Cloud v2 responses: changelog label lists were
verified live against openmrs.atlassian.net (plain label strings, e.g.
'' -> 'intro'). The Claude call itself is exercised by evals/run_evals.py,
which needs credentials; everything else is covered here.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from triage import context as ctx  # noqa: E402
from triage.classifier import LABEL_KEYS, SCHEMA, Classification  # noqa: E402
from triage.jira import JiraClient  # noqa: E402
from triage.run import _load_dotenv, comment_body, load_config, plan_label_writes  # noqa: E402
from triage.state import inspect  # noqa: E402

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


class CommentPaginationTests(unittest.TestCase):
    def test_pages_past_truncated_embedded_list(self):
        class StubResponse:
            status_code = 200
            text = "{}"
            request = type("R", (), {"method": "GET", "url": "stub"})()

            def __init__(self, payload):
                self._payload = payload

            def json(self):
                return self._payload

        class StubSession:
            headers: dict = {}

            def get(self, url, params=None, timeout=None):
                assert params["startAt"] == 2
                assert timeout is not None  # every network call must carry a timeout
                return StubResponse({"comments": [{"body": "third"}], "total": 3})

        client = JiraClient("https://example.invalid")
        client.session = StubSession()
        embedded = {"comments": [{"body": "first"}, {"body": "second"}], "total": 3}
        result = client.comments("O3-1", embedded)
        self.assertEqual([c["body"] for c in result], ["first", "second", "third"])


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
