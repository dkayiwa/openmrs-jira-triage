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

That is proof, and it is also the check that fails silently, because it and the
dev panel rest on the same assumption: that somebody wrote the key down. When
nobody did, both miss together - six of the first nine tickets proposed as
automation candidates already had an open PR and the key search found none of
them. So there is a second, weaker lookup for when the first comes back empty:
search for distinctive wording from the ticket summary (`search_phrases` and
`related_pr_urls`). It recovers about half, and its answer is a suggestion for
a human rather than proof, which is why the two are returned separately and the
caller reports them under separate headings.

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

# A phrase matching more than this many open PRs is describing the project, not
# the ticket. Measured on the launch cohort: "unit tests" returns over 140 open
# PRs, `OpenmrsDatePicker` returned 5 and all were genuinely about it, and the
# exact summary of O3-5801 returns exactly 1 - the PR that fixes it. The first
# two counts track the org's open-PR volume and drift; they are stated loosely
# on purpose, because a figure that reads as exact and is not invites the next
# reader to distrust the numbers beside it.
MAX_PHRASE_HITS = 10

# Each phrase costs one throttled search, so the whole cohort's cost scales with
# this. Identifiers come first because they are the precise ones. Measured over
# the 29-ticket cohort the mean is 1.00 phrase per ticket, well under this cap;
# it is here to bound the tail, not the common case. Setting it to 0 disables
# the content backstop outright without touching the key search - the only way
# to keep proof-based exclusion while declining to guess, since --no-pr-check
# and [github].check_open_prs turn off both.
MAX_PHRASES = 3

# Below this a summary is too short to be distinctive as a quoted phrase.
# Sitting on a real boundary rather than comfortably clear of one: 3 of 100
# sampled O3 summaries are exactly four words ("Update Patient Chart README"),
# so raising this by one silences tickets that are searched today.
MIN_SUMMARY_WORDS = 4

# GitHub rejects a longer search outright: 422 "The search is longer than 256
# characters." Jira allows a 255-character summary, so this is reachable with
# ordinary data, not just hostile data - the longest phrase in today's cohort
# makes a 130-character query, half the budget, and one long summary would
# spend it. Left as a skip rather than a truncation: a truncated phrase is a
# different, looser search whose hits nobody vetted, and quietly widening a
# match is the one thing this backstop must not do.
MAX_QUERY_CHARS = 256

# `useAuditLogs`, `OpenmrsDatePicker`: an internal capital after a lowercase.
_CAMEL = re.compile(r"\b[a-zA-Z][a-z0-9]*(?:[A-Z][a-zA-Z0-9]*){1,}\b")
# `esm-admin-auditlog-app`, `openmrs-module-patientdocuments`: two or more hyphens.
# Case-sensitive, which is a known and measured limit rather than an oversight.
# O3-5752's summary reads "Create frontend smart-notification-App", and the
# capital in `-App` means no identifier is extracted at all - even though
# openmrs-esm-smart-notifications-app is a real repository. Left alone because
# the fix buys nothing here and risks what has already gone wrong once: matching
# case-insensitively would have found `smart-notification-app`, which returns 0
# open PRs, while also admitting capitalised English compounds ("Read-Only-Mode")
# of exactly the kind `weight-for-age` was a false hold-back for.
_KEBAB = re.compile(r"\b[a-z][a-z0-9]*(?:-[a-z0-9]+){2,}\b")

# ...but hyphens alone do not make an identifier, and English supplies plenty of
# compounds that look like one. Measured over the live cohort: `weight-for-age`,
# from "Add CDC weight-for-age growth reference standard", matched an unrelated
# PR of our own that happens to contain the phrase in a test fixture - a false
# hold-back on a ticket nobody was working on. Every real package name in this
# org carries one of these as a segment (openmrs-esm-patient-chart,
# openmrs-module-fhir2, esm-admin-auditlog-app), and no English compound does.
_PACKAGE_MARKERS = frozenset({"openmrs", "esm", "module", "omod", "app"})


class GitHubError(RuntimeError):
    pass


def item_url(item: dict, org: str) -> str:
    """The PR's URL, guaranteed to be a string.

    html_url is what the API returns; url is the API-side fallback. Never
    synthesise "org#123" - it reaches the report as an href and renders as a
    link that goes nowhere, which is worse than plain text a reviewer can
    search for.

    The isinstance check is not defensive habit. These values are untrusted
    remote JSON, and a non-string escapes a long way before anything notices:
    a numeric html_url flows through the client and the journal and only fails
    in write_comment_report, which runs after every classification is paid for,
    so the whole report is lost at the end of an otherwise good sweep. A dict
    or list is worse still - it is unhashable, so it raises TypeError inside
    the client, and TypeError is not GitHubError, so it escapes the handler
    that exists to stop an advisory search from failing a ticket.
    """
    for field in ("html_url", "url"):
        value = item.get(field)
        if isinstance(value, str) and value:
            return value
    return f"{org} PR #{item.get('number')} (no URL returned)"


def searchable_phrases(summary: str, org: str) -> list[tuple[str, str]]:
    """(phrase, query) for each phrase that can actually be sent, in order.

    Separate from search_phrases because "derivable" and "searchable" are not
    the same, and the caller needs the second one. A summary can yield no phrase
    at all ("Fix bug"), or yield phrases whose query exceeds GitHub's 256-char
    limit and so are skipped - and in both cases related_pr_urls returns an
    empty list, which is indistinguishable from "searched and found nothing".

    That gap is the failure this project keeps relearning: a path reporting
    success without the evidence it implies. A ticket the backstop never looked
    at was journalled exactly like one it looked at and cleared, so the audit
    record could not answer "was this ticket in scope?". Asking this function
    lets the caller say which happened without repeating the rules.
    """
    out = []
    for phrase in search_phrases(summary):
        # Quoted only when it is genuinely a phrase. Multi-word text must be
        # quoted or GitHub ORs the words and "Add location name for Transfer
        # Request encounter type" matches anything containing "name". A single
        # identifier must NOT be, and this is not a style choice:
        # `"esm-admin-auditlog-app"` quoted returns 0 results while the same
        # term bare returns the 2 PRs that are the real work, because the index
        # tokenises on the hyphens and the quoted form is matched literally.
        # Quoting everything looked tidier and silently cost a third of what
        # this backstop recovers - it only showed up against live GitHub, never
        # against a stub.
        q = f'"{phrase}"' if " " in phrase else phrase
        query = f"org:{org} is:pr is:open {q}"
        # Dropped rather than sent and rejected. A 422 raises out of the search
        # loop, which would discard the hits already collected from earlier
        # phrases - so one over-long summary would cost a ticket the identifier
        # match that had already succeeded, as well as a throttled search spent
        # to be told the query was malformed.
        if len(query) <= MAX_QUERY_CHARS:
            out.append((phrase, query))
    return out


def search_phrases(summary: str) -> list[str]:
    """Distinctive things to search for, derived from a ticket summary.

    Two kinds, in precision order, established by measuring against the six
    tickets whose in-flight PRs the key search missed:

    Code identifiers are the strong signal. `esm-admin-auditlog-app` from
    O3-5685 finds the PR even though the repository is called
    openmrs-esm-audit-log-app - the identifier survives in the content where
    the repository name does not. `useAuditLogs` from O3-5686 likewise.

    The whole summary as a quoted phrase catches the other shape: a PR whose
    title mirrors the ticket. O3-5801's summary returns exactly one hit, the
    PR that fixes it.

    Nothing here rescues a summary that is ordinary prose with no identifier in
    it - "Add filter bar with entity type, username and date range" returns
    nothing, and no phrasing of it would. That is a real limit of this
    approach, not a tuning problem.
    """
    text = re.sub(r"[{}]", " ", summary or "")
    phrases: list[str] = []
    # Identifiers before the prose summary is a measured ordering: `useAuditLogs`
    # finds O3-5686's PRs and the full summary finds nothing. Kebab before camel
    # is NOT - there is no evidence either shape is the better bet, and it only
    # decides which searches a summary rich in both pays for once MAX_PHRASES
    # bites. It is pinned by a characterisation test so that changing it is a
    # decision rather than an accident, not because this order is known to win.
    for pattern in (_KEBAB, _CAMEL):
        for token in pattern.findall(text):
            if len(token) < 8 or token in phrases:
                continue
            if pattern is _KEBAB and not _PACKAGE_MARKERS.intersection(token.split("-")):
                continue
            phrases.append(token)
    words = re.sub(r"[^\w\s-]", " ", text).split()
    if len(words) >= MIN_SUMMARY_WORDS:
        phrases.append(" ".join(words))
    return phrases[:MAX_PHRASES]


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

    def _request(self, query: str):
        """The search request itself. One definition, so the retry below cannot
        drift from the first attempt.

        Transport failures become GitHubError so that "the search did not
        answer" has exactly one type, whatever the cause. Without this a read
        timeout - seen once in a 34-ticket sweep, so not rare - escapes as
        requests.ReadTimeout, and the caller in run.py that catches GitHubError
        to keep an *advisory* search from failing a ticket does not catch it.
        The transient blip that the wrapper exists to absorb would be the one
        thing that got through it.
        """
        try:
            return self.session.get(SEARCH_URL, params={"q": query, "per_page": 100},
                                    timeout=self.timeout)
        except requests.RequestException as e:
            raise GitHubError(f"GET {SEARCH_URL}?q={query} did not complete: "
                              f"{type(e).__name__}: {e}") from e

    def _search(self, query: str, require_complete: bool = True) -> dict:
        """One throttled search, validated into an answer or a GitHubError.

        The contract both callers depend on: **the only exception this raises is
        GitHubError.** Transport failure, a non-JSON body, a JSON body that is
        not an object, a missing or mistyped field, an item that is not an
        object - all of it arrives as GitHubError, so anything else escaping
        this method means a bug in this module rather than a bad answer from
        GitHub. run.py distinguishes those two on exactly that basis: it absorbs
        GitHubError so an advisory search cannot fail a ticket, and lets every
        other type through so our own bugs stay loud.

        Returns the raw payload; `items` is guaranteed to be a list of dicts and
        `total_count` a non-boolean int, so callers may index them directly.
        """
        self._throttle()
        resp = self._request(query)
        if resp.status_code in (403, 429):
            resp = self._retry_after_rate_limit(resp, query)
        if resp.status_code >= 400:
            raise GitHubError(f"GET {SEARCH_URL}?q={query} -> {resp.status_code}: "
                              f"{resp.text[:300]}")
        # Everything below fails CLOSED, for one reason: this module's job is to
        # prove ABSENCE. A benign-looking empty result is indistinguishable from
        # a real negative, and the consequence of getting it wrong is labelling
        # and commenting on a ticket that is already in review - publicly, and
        # with no way to un-send. A raised error costs one ticket, which the
        # caller journals and the breaker caps.
        if not resp.text:
            raise GitHubError(f"GET {SEARCH_URL}?q={query} -> 200 with an empty body; "
                              "that is no answer, not 'no open PR'")
        # A non-empty body is not a JSON body. The realistic source is not an
        # exotic one: a captive portal or an authenticating proxy answers 200
        # with an HTML page, and 200-with-HTML is the one failure the
        # status_code check above cannot see. json() then raises
        # ValueError/JSONDecodeError, which is not GitHubError, so it escapes
        # the handler that keeps an advisory search from failing a ticket.
        try:
            data = resp.json()
        except ValueError as e:
            raise GitHubError(f"GET {SEARCH_URL}?q={query} -> 200 whose body is not JSON "
                              f"({type(e).__name__}); first bytes: "
                              f"{resp.text[:80]!r}") from e
        # And valid JSON is not necessarily an object. Checked before the first
        # .get() below rather than after: on a bare list, string or number that
        # call raises AttributeError, again escaping as the wrong type.
        if not isinstance(data, dict):
            raise GitHubError(f"GET {SEARCH_URL}?q={query} -> 200 whose body is a JSON "
                              f"{type(data).__name__}, not an object; that is no answer, "
                              "not 'no open PR'")
        # GitHub sets this when its own search timed out. The result set is then
        # not authoritative, so an empty one means "we did not finish looking".
        if data.get("incomplete_results"):
            raise GitHubError(f"GitHub reported incomplete_results for {query}: the "
                              "search timed out, so an empty result is not evidence "
                              "that no open PR names this key")
        # Require the shape before trusting the content. Defaulting these with
        # .get() is how the guard below gets disarmed: `total_count` defaulting
        # to len(items) makes the comparison len(items) > len(items), always
        # false, so any well-formed JSON object that is not a search payload - a
        # proxy envelope, an error body, a future API shape - would sail through
        # all three checks and return an unearned empty.
        if "items" not in data or "total_count" not in data:
            raise GitHubError(f"GET {SEARCH_URL}?q={query} -> 200 whose body is not a "
                              f"search result (keys: {sorted(data)[:6]}); that is no "
                              "answer, not 'no open PR'")
        items = data["items"]
        # Having the keys is not having the shape. The guard above accepts any
        # body carrying both names, so a proxy envelope or an error page shaped
        # as JSON still reaches here - and then `item.get(...)` raises
        # AttributeError and `total_count > len(items)` raises TypeError, which
        # are NOT GitHubError.
        #
        # That distinction is the whole point. run.py catches GitHubError to
        # keep the advisory content search from failing a ticket, and lets
        # anything else through so a real coding bug in this module errors
        # loudly rather than hiding as "the second look did not happen". A
        # malformed remote response is untrusted input, not a bug in us, so it
        # has to arrive as GitHubError - otherwise a bad gateway turns a
        # graceful degradation into five errored tickets and a tripped breaker.
        if not isinstance(items, list) or not all(isinstance(i, dict) for i in items):
            raise GitHubError(f"GET {SEARCH_URL}?q={query} -> 200 whose 'items' is not a "
                              f"list of objects ({type(items).__name__}); that is no "
                              "answer, not 'no open PR'")
        if not isinstance(data["total_count"], int) or isinstance(data["total_count"], bool):
            raise GitHubError(f"GET {SEARCH_URL}?q={query} -> 200 whose 'total_count' is "
                              f"{data['total_count']!r}, not a number; that is no answer, "
                              "not 'no open PR'")
        # We do not paginate - one key should never match more than a page of
        # PRs. If it somehow does, the unseen pages could hold the match, and
        # silently answering from the first page would be a false negative.
        # Truncation is fatal to a key search and merely informative to a phrase
        # search, so the caller says which it is. Proving absence cannot survive
        # unseen pages: the match could be on one of them. Judging whether a
        # phrase is distinctive only needs the count, and a phrase that overflows
        # a page has already answered - "unit tests" returns over 140 open PRs, which
        # is the definition of not distinctive. Failing there would turn the
        # cheapest possible "no" into an error that aborts the remaining phrases
        # for that ticket, so the guard would fire hardest on the phrases it was
        # written to dismiss.
        if require_complete and data["total_count"] > len(items):
            raise GitHubError(f"GitHub reported {data['total_count']} matches for "
                              f"{query} but "
                              f"returned {len(items)}; this client does not paginate, "
                              "so the remainder cannot be ruled out")
        return data

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
        return self._request(query)

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

    def related_pr_urls(self, key: str, summary: str) -> list[tuple[str, str]]:
        """Open PRs that look like this ticket's work but never cite its key.

        The key search below is the authoritative signal and it is also the one
        that fails. Measured on the launch cohort: six of the nine tickets the
        sweep proposed as automation candidates already had an open PR, and the
        key search found NONE of them, because not one cited its key in the
        title or body. Jira's dev panel missed them for the same reason, so the
        two mechanisms the pilot calls independent are the same question asked
        twice - one citation gap defeats both.

        It is not only forgetfulness. openmrs-esm-audit-log-app#1 does cite a
        key now, O3-5843 - the epic - while the five tickets describing that
        work are O3-5685..5689. A PR that cites the parent is invisible to a
        search for the child.

        So this asks a different question: does an open PR describe the same
        work? Returns (url, evidence) pairs, where evidence is the phrase that
        matched, because a content match is a suggestion and the reader needs
        to see what it rested on. Callers must keep it distinct from a key
        match, which is proof.

        Intended to be called only when open_pr_urls came back empty. Nothing
        breaks if it is not, but the reasoning below about keeping key-naming
        PRs assumes it, and a caller that ignores that will see the same PR
        twice.
        """
        found: dict[str, str] = {}
        # Which phrases are worth sending, and how, is searchable_phrases'
        # business - shared with the caller so it can tell "never searched" from
        # "searched and clear" without duplicating the rules.
        for phrase, query in searchable_phrases(summary, self.org):
            data = self._search(query, require_complete=False)
            total = data["total_count"]
            # A phrase that matches half the org is not evidence of anything.
            # Measured: "unit tests" returns over 140 open PRs, while the exact
            # summary of O3-5801 returns exactly one - the PR that fixes it.
            # The cutoff is on the ANSWER rather than on the phrase, because
            # judging distinctiveness from the words alone means guessing at a
            # corpus this can simply consult.
            if total > MAX_PHRASE_HITS:
                continue
            for item in data["items"]:
                # A PR that turns out to name the key is KEPT, not filtered out
                # as "the key search already has it". It demonstrably does not:
                # the only caller runs this when open_pr_urls returned nothing,
                # so a key-naming PR reaching here is one the key search missed.
                # Dropping it would discard proof-grade evidence and let the
                # sweep label a ticket that is provably in review - the worst
                # outcome this pipeline has. A filter that cannot deduplicate,
                # because by construction there is nothing to deduplicate
                # against, can only ever lose data.
                url = item_url(item, self.org)
                cited = f"{item.get('title') or ''}\n{item.get('body') or ''}"
                # Said plainly in the evidence, because it changes what the
                # match is worth: this one is not a resemblance, it is a
                # citation the key search failed to see.
                found.setdefault(url, f"{phrase} (and names {key})"
                                 if names_key(cited, key) else phrase)
        return sorted(found.items())

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
                urls.append(item_url(item, self.org))
        return urls
