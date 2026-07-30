"""Open-PR lookup: the backstop for Jira's dev panel.

`scope_jql` drops any ticket whose dev panel lists a pull request, but that
panel only knows what the GitHub-for-Jira app indexed. Measured on the launch
cohort: openmrs-esm-core#1818 is open and its title reads "(fix) O3-5816: Stop
LocationPicker crashing when no login locations are configured", yet
`development[pullrequests].all = 0` still matched O3-5816 - so the sweep
offered a ticket that was already in review. The clause itself is not broken
and does evaluate anonymously (2377 O3 tickets match `.all > 0`); the gap is
the index, not the JQL, and no JQL edit can close it.

So this asks GitHub for the same evidence the dev panel is supposed to carry:
an open PR naming the ticket key. A ticket already in review must not be
labelled, because the maintainer's removal of that label is a permanent opt-out
and counts against the pilot's kill metric.

One search per key, not one org-wide listing: openmrs has ~1600 open PRs and
the search API truncates at 1000 results, so "fetch them all and match locally"
cannot be made correct.
"""
from __future__ import annotations

import re
import time

import requests

SEARCH_URL = "https://api.github.com/search/issues"

# Search's own rate limits, which are per-minute and far tighter than the REST
# core limits. Staying under them by construction beats discovering them at
# ticket 11 of 34 and failing the rest of the sweep.
RATE_LIMIT_PER_MIN = {True: 30, False: 10}

# Pacing at exactly the documented rate is pacing at the limit: measured on a
# real 32-ticket sweep, a 2.0s interval (30/min authenticated, the documented
# maximum) took a 403 on the last ticket, because anything else sharing the
# token - a `gh` invocation, a previous sweep's tail - lands in the same window.
# Aim below the ceiling so the sweep has somewhere to drift.
RATE_HEADROOM = 0.8

# One bounded wait when the limit is hit anyway (clock skew, a shared runner
# IP), then fail loudly. Sleeping through a long reset would silently stall a
# sweep the pilot expects to finish inside its 24h SLA.
MAX_RATE_WAIT_SECONDS = 75

# The search window is a minute, so this is the wait when GitHub says "rate
# limit exceeded" without a header saying when it reopens - the secondary-limit
# response, which carries neither retry-after nor an exhausted remaining count.
DEFAULT_RATE_WAIT_SECONDS = 60

class GitHubError(RuntimeError):
    pass


def names_key(text: str, key: str) -> bool:
    """Does `text` cite this exact issue key?

    Bounded on both sides so O3-581 does not match a PR about O3-5816, and
    O3-5816 does not match O3-58161. Case-insensitive because Jira accepts and
    normalises a lowercase key, so a PR title may carry either form.
    """
    pattern = rf"(?<![0-9A-Za-z]){re.escape(key)}(?![0-9A-Za-z])"
    return re.search(pattern, text or "", re.IGNORECASE) is not None


class GitHubClient:
    """Searches one org for open PRs naming a ticket key.

    `sleep` and `now` are injected so the throttle is testable without real
    time passing.
    """

    def __init__(self, org: str, token: str | None = None, timeout: int = 30,
                 sleep=time.sleep, now=time.monotonic):
        self.org = org
        self.timeout = timeout
        self.authenticated = bool(token)
        self._sleep = sleep
        self._now = now
        self._last_request: float | None = None
        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        })
        if token:
            self.session.headers["Authorization"] = f"Bearer {token}"

    @property
    def min_interval(self) -> float:
        return 60.0 / (RATE_LIMIT_PER_MIN[self.authenticated] * RATE_HEADROOM)

    def _throttle(self) -> None:
        if self._last_request is not None:
            wait = self.min_interval - (self._now() - self._last_request)
            if wait > 0:
                self._sleep(wait)
        self._last_request = self._now()

    def _search(self, query: str) -> dict:
        self._throttle()
        resp = self.session.get(SEARCH_URL, params={"q": query, "per_page": 100},
                                timeout=self.timeout)
        if resp.status_code in (403, 429):
            resp = self._retry_after_rate_limit(resp, query)
        if resp.status_code >= 400:
            raise GitHubError(f"GET {SEARCH_URL}?q={query} -> {resp.status_code}: "
                              f"{resp.text[:300]}")
        return resp.json() if resp.text else {}

    def _retry_after_rate_limit(self, resp, query: str):
        """Wait out a short rate-limit window once, else fail with the reset.

        Fails loudly rather than proceeding: a swallowed rate limit would report
        "no open PR" for every remaining ticket, which is exactly the wrong
        answer - it re-opens the leak this module exists to close, silently, and
        on the tickets most likely to be in review.
        """
        wait = self._retry_delay(resp)
        if wait is None or wait > MAX_RATE_WAIT_SECONDS:
            raise GitHubError(
                f"GitHub search rate limit reached ({resp.status_code}) and the "
                f"window does not reopen for {'an unknown time' if wait is None else f'{wait:.0f}s'}. "
                "Set GITHUB_TOKEN to raise the limit from 10 to 30 searches/min, "
                "or pass --no-pr-check to sweep without the dev-panel backstop."
            )
        self._sleep(wait)
        self._last_request = self._now()
        return self.session.get(SEARCH_URL, params={"q": query, "per_page": 100},
                                timeout=self.timeout)

    @staticmethod
    def _retry_delay(resp) -> float | None:
        """Seconds until the window reopens, or None if this is not a rate limit.

        Three shapes, because GitHub sends three. `retry-after` is explicit.
        An exhausted `x-ratelimit-remaining` pairs with a reset timestamp. And
        the secondary limit sends neither - just a 403 whose body says the limit
        was exceeded - which is the one a real 32-ticket sweep actually hit;
        reading that as "not a rate limit" is why it failed the ticket instead of
        waiting a minute for the window to reopen.
        """
        headers = resp.headers or {}
        retry_after = headers.get("retry-after")
        if retry_after:
            try:
                return max(0.0, float(retry_after))
            except ValueError:
                pass
        if headers.get("x-ratelimit-remaining") in ("0", 0):
            reset = headers.get("x-ratelimit-reset")
            try:
                return max(0.0, float(reset) - time.time())
            except (TypeError, ValueError):
                return DEFAULT_RATE_WAIT_SECONDS
        # No usable header. Believe the body, but only when it says so: a 403 for
        # any other reason (a bad token, SAML enforcement) must not be read as a
        # wait, or every such failure costs a pointless minute per ticket.
        body = (getattr(resp, "text", "") or "").lower()
        if "rate limit" in body or "secondary rate limit" in body:
            return DEFAULT_RATE_WAIT_SECONDS
        return None

    def open_pr_urls(self, key: str) -> list[str]:
        """URLs of open PRs whose title or body names `key`.

        The search itself is full-text and also matches PR *comments*, which is
        wider than the dev panel's notion of a link - "unrelated to O3-5816" in
        a review comment would exclude a ticket nobody is working on. So results
        are re-checked against the title and body only, the two fields a PR
        author uses to claim a ticket.
        """
        data = self._search(f"org:{self.org} is:pr is:open {key}")
        urls = []
        for item in data.get("items", []):
            cited = f"{item.get('title') or ''}\n{item.get('body') or ''}"
            if names_key(cited, key):
                urls.append(item.get("html_url") or f"{self.org}#{item.get('number')}")
        return urls
