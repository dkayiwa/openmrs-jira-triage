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
            keys += [i["key"] for i in data.get("issues", [])]
            token = data.get("nextPageToken")
            if not token:
                return keys

    def _page_rest(self, path: str, item_key: str, out: list[dict], total: int) -> list[dict]:
        """Append pages of `path` to `out` until it holds `total` items.

        Jira caps embedded sub-resources (the issue GET's first page of
        comments or changelog entries) at 100, so anything longer must be
        completed from the dedicated endpoint.
        """
        while len(out) < total:
            data = self._check(self._get(path, params={"startAt": len(out), "maxResults": 100}))
            # /comment wraps items in "comments"; /changelog wraps them in
            # "values" (both verified live against openmrs.atlassian.net).
            page = data.get(item_key) or data.get("values") or []
            if not page:
                break
            out += page
            total = data.get("total", total)
        return out

    def comments(self, key: str, embedded: dict | None) -> list[dict]:
        """Full comment list, paging past the issue GET's embedded first page.

        The visible-information contract includes every human comment, so a
        chatty ticket must not silently lose its newest comments to the
        embedded page limit.
        """
        emb = embedded or {}
        out = list(emb.get("comments", []))
        return self._page_rest(f"/rest/api/2/issue/{key}/comment", "comments",
                               out, emb.get("total", len(out)))

    def changelog(self, key: str, embedded: dict | None) -> list[dict]:
        """Full changelog history, paging past the issue GET's embedded page.

        The changelog is the opt-out state store and expand=changelog embeds
        only the first 100 entries - so on a heavily-edited ticket the entries
        dropped are the *newest*, which is exactly where an opt-out removal
        lives. Missing one would re-label a ticket a human opted out of.
        """
        emb = embedded or {}
        out = list(emb.get("histories", []))
        return self._page_rest(f"/rest/api/2/issue/{key}/changelog", "histories",
                               out, emb.get("total", len(out)))

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
