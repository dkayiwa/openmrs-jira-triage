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
    def __init__(self, base_url: str, email: str | None = None, api_token: str | None = None):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers["Accept"] = "application/json"
        if email and api_token:
            self.session.auth = (email, api_token)
        self.authenticated = bool(email and api_token)

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def _check(self, resp: requests.Response) -> dict | list:
        if resp.status_code >= 400:
            raise JiraError(
                f"{resp.request.method} {resp.request.url} -> {resp.status_code}: {resp.text[:500]}"
            )
        return resp.json() if resp.text else {}

    # -- reads -------------------------------------------------------------

    def server_info(self) -> dict:
        return self._check(self.session.get(self._url("/rest/api/2/serverInfo")))

    def myself(self) -> dict | None:
        resp = self.session.get(self._url("/rest/api/2/myself"))
        return resp.json() if resp.status_code == 200 else None

    def fields(self) -> list[dict]:
        return self._check(self.session.get(self._url("/rest/api/2/field")))

    def project_statuses(self, project: str) -> list[dict]:
        return self._check(self.session.get(self._url(f"/rest/api/2/project/{project}/statuses")))

    def search_keys(self, jql: str) -> list[str]:
        """All issue keys matching jql (POST /rest/api/3/search/jql, GET fallback)."""
        keys: list[str] = []
        token: str | None = None
        while True:
            body: dict = {"jql": jql, "fields": ["key"], "maxResults": 100}
            if token:
                body["nextPageToken"] = token
            resp = self.session.post(self._url("/rest/api/3/search/jql"), json=body)
            if resp.status_code in (401, 403, 405):
                params = {"jql": jql, "fields": "key", "maxResults": 100}
                if token:
                    params["nextPageToken"] = token
                resp = self.session.get(self._url("/rest/api/3/search/jql"), params=params)
            data = self._check(resp)
            keys += [i["key"] for i in data.get("issues", [])]
            token = data.get("nextPageToken")
            if not token:
                return keys

    def issue(self, key: str, fields: list[str], expand_changelog: bool = False) -> dict:
        # expand=changelog embeds up to the first 100 history entries, plenty
        # for young pilot tickets; page /changelog directly if that ever grows.
        params = {"fields": ",".join(fields)}
        if expand_changelog:
            params["expand"] = "changelog"
        return self._check(self.session.get(self._url(f"/rest/api/2/issue/{key}"), params=params))

    # -- writes (live mode only) --------------------------------------------

    def update_labels(self, key: str, add: list[str], remove: list[str]) -> None:
        ops = [{"add": l} for l in add] + [{"remove": l} for l in remove]
        body = {"update": {"labels": ops}}
        # notifyUsers=false needs project-admin; fall back to a notifying edit.
        resp = self.session.put(
            self._url(f"/rest/api/2/issue/{key}"), params={"notifyUsers": "false"}, json=body
        )
        if resp.status_code == 403:
            resp = self.session.put(self._url(f"/rest/api/2/issue/{key}"), json=body)
        self._check(resp)

    def add_comment(self, key: str, body: str) -> dict:
        return self._check(
            self.session.post(self._url(f"/rest/api/2/issue/{key}/comment"), json={"body": body})
        )

    def get_property(self, key: str, prop: str) -> dict | None:
        resp = self.session.get(self._url(f"/rest/api/2/issue/{key}/properties/{prop}"))
        return resp.json().get("value") if resp.status_code == 200 else None

    def set_property(self, key: str, prop: str, value: dict) -> None:
        resp = self.session.put(
            self._url(f"/rest/api/2/issue/{key}/properties/{prop}"), json=value
        )
        if resp.status_code >= 400:
            raise JiraError(f"set_property {key} -> {resp.status_code}: {resp.text[:300]}")
