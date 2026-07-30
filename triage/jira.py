"""Minimal Jira Cloud REST client for the triage pilot.

Reads use API v2, which returns descriptions and comments as wiki-markup
strings (v3 returns ADF JSON that would need walking). The scope sweep uses
the newer /rest/api/3/search/jql endpoint; the old /search was retired on
Cloud. Read paths work anonymously on public projects; writes need basic auth
(bot email + API token).
"""
from __future__ import annotations

import requests


class JiraError(RuntimeError):
    pass


class JiraClient:
    def __init__(self, base_url: str, email: str | None = None, api_token: str | None = None,
                 timeout: int = 30):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout  # a hung connection must fail a ticket, not wedge the sweep
        self.session = requests.Session()
        self.session.headers["Accept"] = "application/json"
        if email and api_token:
            self.session.auth = (email, api_token)
        self.authenticated = bool(email and api_token)

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def _get(self, path: str, params: dict | None = None) -> requests.Response:
        return self.session.get(self._url(path), params=params, timeout=self.timeout)

    def _check(self, resp: requests.Response) -> dict | list:
        if resp.status_code >= 400:
            raise JiraError(
                f"{resp.request.method} {resp.request.url} -> {resp.status_code}: {resp.text[:500]}"
            )
        return resp.json() if resp.text else {}

    # -- reads -------------------------------------------------------------

    def server_info(self) -> dict:
        return self._check(self._get("/rest/api/2/serverInfo"))

    def myself(self) -> dict | None:
        resp = self._get("/rest/api/2/myself")
        return resp.json() if resp.status_code == 200 else None

    def fields(self) -> list[dict]:
        return self._check(self._get("/rest/api/2/field"))

    def project_statuses(self, project: str) -> list[dict]:
        return self._check(self._get(f"/rest/api/2/project/{project}/statuses"))

    def search_keys(self, jql: str) -> list[str]:
        """All issue keys matching jql (POST /rest/api/3/search/jql, GET fallback)."""
        keys: list[str] = []
        token: str | None = None
        seen_tokens: set[str] = set()
        while True:
            body: dict = {"jql": jql, "fields": ["key"], "maxResults": 100}
            if token:
                body["nextPageToken"] = token
            resp = self.session.post(self._url("/rest/api/3/search/jql"), json=body,
                                     timeout=self.timeout)
            if resp.status_code in (401, 403, 405):
                params = {"jql": jql, "fields": "key", "maxResults": 100}
                if token:
                    params["nextPageToken"] = token
                resp = self._get("/rest/api/3/search/jql", params=params)
            data = self._check(resp)
            # An empty cohort and a body that is not a search result look the
            # same through .get("issues", []): both yield zero keys, and zero
            # keys is a sweep that does nothing and reports success. Jira
            # returns "issues" even for no matches, so its absence means the
            # response is something else - a warningMessages-only body from a
            # JQL field Jira does not know, an error envelope, a proxy page.
            if "issues" not in data:
                raise JiraError(
                    f"search/jql returned 200 whose body is not a search result "
                    f"(keys: {sorted(data)[:6]}); that is no answer, not an empty cohort")
            keys += [i["key"] for i in data["issues"]]
            token = data.get("nextPageToken")
            if not token:
                return keys
            # A token Jira has already handed us means the cursor is not moving.
            # Left unguarded that is an infinite loop: the sweep never finishes,
            # never reports, and burns the rate limit until the job is killed.
            if token in seen_tokens:
                raise JiraError(f"search/jql repeated nextPageToken {token!r} after "
                                f"{len(keys)} keys; the cursor is not advancing")
            seen_tokens.add(token)

    def _complete(self, path: str, embedded: dict | None, embedded_key: str,
                  page_key: str) -> list[dict]:
        """Every item of a sub-resource, cheaply when the issue GET sufficed.

        The embedded page cannot be extended, only replaced. Jira returns the
        *newest* window there, and does not describe it consistently - verified
        live: LUI-45's embedded comments report startAt 35 of total 135, while
        TRUNK-324's embedded changelog reports startAt 0 yet returns the newest
        100 in descending order, opposite to the dedicated endpoint's ascending
        order. Appending dedicated pages to that window therefore duplicates
        the overlap and silently drops the oldest entries. So the embedded page
        is used only when it is already complete; once truncated it is
        discarded and the dedicated endpoint is read from the start.
        """
        emb = embedded or {}
        # "No embedded page" is not "no items". An absent page - the caller did
        # not request the field, or Jira dropped it - carries no total, so
        # defaulting the total to the zero items in hand certifies an emptiness
        # that was never obtained: the check below compares 0 >= 0 and returns
        # a complete-looking answer without one request. Unknown means go and
        # ask. (Same disarming default as a search whose total_count falls back
        # to len(items), which github.py refuses for the same reason.)
        if embedded is not None and "total" in emb:
            items = list(emb.get(embedded_key, []))
            if len(items) >= emb["total"]:
                return items
        items = []
        total = None
        while True:
            data = self._check(self._get(path, params={"startAt": len(items),
                                                       "maxResults": 100}))
            page = data.get(page_key) or []
            items += page
            if "total" in data:
                total = data["total"]
            # Everything below fails CLOSED, because this helper's callers read
            # its result as exhaustive: changelog() decides whether a human
            # opted a ticket out, and comments() decides whether a question has
            # already been answered. A short read is indistinguishable from a
            # complete one, and answering "no opt-out" from history nobody
            # fetched re-labels a ticket a maintainer opted out of - permanently
            # and publicly. Raising costs one ticket, which the caller journals.
            if total is None:
                raise JiraError(
                    f"GET {path} returned no total, so there is no way to tell "
                    f"whether these {len(items)} entries are all of them")
            if len(items) >= total:
                return items
            if not page:
                # Guards the loop against a server that never advances, but the
                # bound must not be paid for by silently truncating: returning
                # here is exactly the short read described above.
                raise JiraError(
                    f"GET {path} reported {total} entries but stopped returning "
                    f"them at {len(items)}; the remainder cannot be ruled out")

    def comments(self, key: str, embedded: dict | None) -> list[dict]:
        """Every comment on the ticket.

        The visible-information contract covers every human comment, so a
        chatty ticket must not silently lose any to the embedded page limit.
        """
        return self._complete(f"/rest/api/2/issue/{key}/comment", embedded,
                              "comments", "comments")

    def changelog(self, key: str, embedded: dict | None) -> list[dict]:
        """Every changelog entry on the ticket.

        This is the opt-out state store: a missed non-bot label removal would
        re-label a ticket a human opted out of. The bot's own first label add
        (the 24h SLA's start point) is likewise only findable in the full
        history. The dedicated endpoint wraps entries in "values", not
        "histories".
        """
        return self._complete(f"/rest/api/2/issue/{key}/changelog", embedded,
                              "histories", "values")

    def issue(self, key: str, fields: list[str], expand_changelog: bool = False) -> dict:
        # expand=changelog embeds only the first 100 history entries; callers
        # that must not miss one complete it via changelog() below.
        params = {"fields": ",".join(fields)}
        if expand_changelog:
            params["expand"] = "changelog"
        issue = self._check(self._get(f"/rest/api/2/issue/{key}", params=params))
        if expand_changelog and "changelog" not in issue:
            # A ticket with no history still comes back with an empty changelog
            # object, so an absent key means the expansion was not honoured. It
            # must not be read as "never touched": that would make every
            # opt-out invisible and re-label tickets humans opted out of.
            raise JiraError(f"{key}: expand=changelog returned no changelog")
        return issue

    # -- writes (live mode only) --------------------------------------------

    def update_labels(self, key: str, add: list[str], remove: list[str]) -> None:
        ops = [{"add": l} for l in add] + [{"remove": l} for l in remove]
        body = {"update": {"labels": ops}}
        # notifyUsers=false needs project-admin; fall back to a notifying edit.
        resp = self.session.put(self._url(f"/rest/api/2/issue/{key}"),
                                params={"notifyUsers": "false"}, json=body,
                                timeout=self.timeout)
        if resp.status_code == 403:
            resp = self.session.put(self._url(f"/rest/api/2/issue/{key}"), json=body,
                                    timeout=self.timeout)
        self._check(resp)

    def add_comment(self, key: str, body: str) -> dict:
        return self._check(self.session.post(self._url(f"/rest/api/2/issue/{key}/comment"),
                                             json={"body": body}, timeout=self.timeout))

    def delete_comment(self, key: str, comment_id: str) -> None:
        # Only preflight uses this, to clean up its permission probe. The pilot
        # itself never deletes anything.
        self._check(self.session.delete(
            self._url(f"/rest/api/2/issue/{key}/comment/{comment_id}"), timeout=self.timeout))

    def delete_property(self, key: str, prop: str) -> None:
        self._check(self.session.delete(
            self._url(f"/rest/api/2/issue/{key}/properties/{prop}"), timeout=self.timeout))

    def get_property(self, key: str, prop: str) -> dict | None:
        # 404 means "never triaged" (the normal first-run case); anything else
        # must fail loudly - swallowing a 401/500 here would silently reclassify
        # every ticket on every live run instead of reporting broken auth.
        resp = self._get(f"/rest/api/2/issue/{key}/properties/{prop}")
        if resp.status_code == 404:
            return None
        return (self._check(resp) or {}).get("value")

    def set_property(self, key: str, prop: str, value: dict) -> None:
        resp = self.session.put(self._url(f"/rest/api/2/issue/{key}/properties/{prop}"),
                                json=value, timeout=self.timeout)
        if resp.status_code >= 400:
            raise JiraError(f"set_property {key} -> {resp.status_code}: {resp.text[:300]}")
