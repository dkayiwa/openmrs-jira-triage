"""Pre-launch checks from the pilot's Jira-setup section.

Verifies connectivity/auth, that the scope JQL runs (including the
development[] clause), that the Acceptance Criteria field and "To Do" status
exist, that the GitHub open-PR backstop can run the same search the sweep will,
and - with --scratch and bot credentials - that Jira really rejects '/' in
labels, which is why config.toml uses hyphenated ai-triage-* names instead of
the design doc's ai-triage/* form.
"""
from __future__ import annotations

import argparse
import os
import sys

from anthropic import Anthropic

from . import context as ctx
from .run import bot_identity_error, github_from_env, jira_from_env, load_config
from .state import PROPERTY_KEY


def check(label: str, ok: bool, detail: str = "") -> bool:
    print(f"[{'PASS' if ok else 'FAIL'}] {label}" + (f" - {detail}" if detail else ""))
    return ok


def attempt(label: str, fn):
    """Run one probe; return (ok, value). A raised exception becomes a FAIL line.

    Preflight's whole job is to report which checks pass, so letting one of them
    abort the run hides every check after it - precisely when the operator most
    needs the full picture. A 404 on project statuses used to mean you never
    learned about the Acceptance Criteria field, the scope JQL or the bot id.
    """
    try:
        return True, fn()
    except Exception as e:
        check(label, False, f"{type(e).__name__}: {e}"[:200])
        return False, None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="triage pilot preflight")
    ap.add_argument("--scratch", help="issue key for the label-charset write test (needs bot auth)")
    args = ap.parse_args(argv)

    cfg = load_config()
    jira = jira_from_env(cfg)
    ok = True

    try:
        info = jira.server_info()
    except Exception as e:
        check("jira reachable", False, str(e)[:200])
        return 1
    ok &= check("jira reachable", True, f"{info.get('baseUrl')} ({info.get('deploymentType')})")
    me = jira.myself()
    print(f"       auth: {'as ' + (me.get('displayName') or '?') if me else 'anonymous (read-only)'}")

    # Checked here because a wrong bot id is silent at runtime: the bot's own
    # label flips would read as human opt-outs and permanently skip tickets.
    bot_id = os.environ.get("TRIAGE_BOT_ACCOUNT_ID")
    if not bot_id:
        print("       TRIAGE_BOT_ACCOUNT_ID: unset (required for --live)")
    elif not jira.authenticated:
        print(f"       TRIAGE_BOT_ACCOUNT_ID: {bot_id} (unverified - no bot credentials)")
    else:
        mismatch = bot_identity_error(jira, bot_id)
        ok &= check("TRIAGE_BOT_ACCOUNT_ID matches credentials", not mismatch, mismatch or bot_id)

    # Reads the configured status rather than a literal, so renaming the pilot's
    # in-scope status cannot leave preflight validating one the pipeline no
    # longer uses while every live ticket is skipped as out of scope.
    scope_status = cfg["jira"]["scope_status"]
    status_label = f'"{scope_status}" status exists'
    got, itypes = attempt(status_label,
                          lambda: jira.project_statuses(cfg["jira"]["project"]))
    if got:
        statuses = {s["name"] for itype in itypes for s in itype["statuses"]}
        ok &= check(status_label, scope_status in statuses,
                    f"project statuses: {sorted(statuses)}")
    else:
        ok = False

    # Deliberately not `ok = ok and got and check(...)`: `and` short-circuits, so
    # once an earlier probe has failed the check would never run and its line
    # would vanish from the report - the very fault this restructuring fixes.
    got, ac = attempt("Acceptance Criteria field", lambda: ctx.discover_ac_field(
        jira, cfg["jira"].get("acceptance_criteria_field", "")))
    if got:
        ok &= check("Acceptance Criteria field", bool(ac), ac or "not found")
    else:
        ok = False

    jql = cfg["jira"]["scope_jql"].format(since=cfg["jira"]["cohort_created_since"])
    dev_clause = cfg["jira"]["dev_panel_clause"]
    scope_label = "scope JQL (with development[] clause)"
    scope_ok, keys = attempt(scope_label, lambda: jira.search_keys(jql))
    if scope_ok and keys:
        ok &= check(scope_label, True, f"{len(keys)} ticket(s); doc estimates ~35")
    elif scope_ok:
        # An empty result is not an error, so attempt() calls it a success - but
        # a development[] clause Jira cannot evaluate (no GitHub-for-Jira app, or
        # no dev-panel access) returns an empty set rather than a 400. Reporting
        # PASS on that means the gate certifies a sweep that will find nothing.
        # The clause-free count tells the two apart: both zero is a genuinely
        # empty cohort; a difference is the clause silently not being applied.
        bare_ok, bare = attempt("scope JQL without the development[] clause",
                                lambda: jira.search_keys(jql.replace(dev_clause, "")))
        # Three states, not two. attempt() prints its own FAIL line for a control
        # query that raised but does not touch `ok`, so folding that case in with
        # "both empty" would assert a clause-free count of zero that was never
        # obtained - the same unearned PASS, surviving in the error state.
        if not bare_ok:
            ok &= check(scope_label, False,
                        "0 ticket(s), and the clause-free control query failed too, "
                        "so there is no evidence either way about whether the cohort "
                        "is empty or the development[] clause is being ignored")
        elif bare:
            ok &= check(scope_label, False,
                        f"0 ticket(s) with the development[] clause but {len(bare)} "
                        "without it - the clause is not being evaluated here, so "
                        "the sweep would see an empty cohort")
        else:
            ok &= check(scope_label, True,
                        "0 ticket(s), and 0 without the development[] clause too - "
                        "the cohort is genuinely empty")
    else:
        ok = False
        fallback_ok, fallback = attempt("scope JQL without the development[] clause",
                                lambda: jira.search_keys(jql.replace(dev_clause, "")))
        if fallback_ok:
            print(f"       without the clause: {len(fallback)} ticket(s) - the dev-panel "
                  "JQL may need auth or the GitHub-for-Jira app")

    # The dev-panel backstop. Probed with a single search rather than the whole
    # cohort: unauthenticated search allows 10/min, so sweeping every in-scope
    # key here would take minutes and teach nothing the sweep will not report.
    # Built by the same factory the sweep uses, so preflight cannot pass against a
    # differently-configured backstop than the one that will run.
    gh = github_from_env(cfg)
    if gh is None:
        print("       open-PR backstop: disabled in config.toml; scope rests on Jira's "
              "dev panel alone")
    else:
        gh_label = "github open-PR backstop"
        # A key the sweep will actually ask about, so a probe that passes proves
        # the query the pipeline runs - not a simpler one.
        real_key = bool(scope_ok and keys)
        probe_key = keys[0] if real_key else cfg["jira"]["project"] + "-1"
        probed, urls = attempt(gh_label, lambda: gh.open_pr_urls(probe_key))
        if probed:
            auth = "GITHUB_TOKEN" if gh.authenticated else \
                f"unauthenticated: {gh.min_interval:.0f}s/search, set GITHUB_TOKEN to cut it"
            # Say so when the key is made up: the probe still proves the
            # search works, but not against anything the sweep will ask for.
            provenance = "" if real_key else " (synthetic - the scope query returned nothing)"
            ok &= check(gh_label, True, f"org {gh.org}, searched {probe_key}"
                                        f"{provenance} ({len(urls)} open PR(s)); {auth}")
        else:
            ok = False
            print("       the sweep fails a ticket rather than classifying it when this "
                  "search fails; pass --no-pr-check to sweep without the backstop")

    if args.scratch and jira.authenticated:
        # The hyphenated add runs FIRST and gates the slash conclusion. Without
        # it, any Jira failure at all - a 404 from a mistyped --scratch key, a
        # missing Edit Issues permission, a 5xx - reads as "slash rejected" and
        # the go-live gate reports PASS on a probe that tested nothing.
        hyphen_label = "ai-triage-charset-test"
        hyphen_ok = False
        try:
            jira.update_labels(args.scratch, [hyphen_label], [])
            hyphen_ok = True
            ok &= check("hyphenated label accepted", True, hyphen_label)
        except Exception as e:
            ok &= check("hyphenated label accepted", False, str(e)[:200])
        # Only clean up what actually landed, and catch broadly: a transport
        # error is not a JiraError, and letting one escape would abort
        # preflight before the comment and property probes while leaving the
        # label behind.
        if hyphen_ok:
            try:
                jira.update_labels(args.scratch, [], [hyphen_label])
            except Exception as e:
                print(f"       WARN: could not remove {hyphen_label} from "
                      f"{args.scratch} ({type(e).__name__}); remove manually")

        slash_label = "ai-triage/charset-test"
        # Only the *add* may count as evidence: wrapping the cleanup removal in
        # the same try would report a failed removal as "slash rejected", the
        # opposite conclusion, and leave the label on the scratch ticket.
        try:
            jira.update_labels(args.scratch, [slash_label], [])
        except Exception as e:
            if hyphen_ok:
                check("slash rejected in labels", True,
                      "hyphenated ai-triage-* names are required")
            else:
                ok &= check("slash rejected in labels", False,
                            "inconclusive - the hyphenated add failed too, so this "
                            f"is not evidence about '/': {str(e)[:120]}")
        else:
            check("slash rejected in labels", False,
                  f"Jira accepted {slash_label!r}; the doc's ai-triage/* names would work after all")
            ok = False
            try:
                jira.update_labels(args.scratch, [], [slash_label])
            except Exception as e:
                print(f"       WARN: could not remove {slash_label} from "
                      f"{args.scratch} ({type(e).__name__})")
        # Add Comments is probed explicitly because its absence produces the one
        # failure the pipeline cannot recover from: labels are written before
        # comments, so a missing permission leaves the ticket labelled with no
        # explanation, and no later run will add one (the label's presence
        # suppresses the comment).
        posted = None
        try:
            posted = jira.add_comment(args.scratch, "triage pilot preflight - "
                                      "verifying Add Comments; this will be deleted")
            ok &= check("bot can add comments", True, f"comment {posted.get('id')}")
        except Exception as e:
            ok &= check("bot can add comments", False, str(e)[:200])
        finally:
            if posted and posted.get("id"):
                try:
                    jira.delete_comment(args.scratch, posted["id"])
                except Exception as e:
                    print(f"       WARN: could not delete preflight comment "
                          f"{posted['id']} from {args.scratch} "
                          f"({type(e).__name__}); remove manually")

        # Entity properties are how the pipeline remembers what it has triaged.
        # Without this permission every ticket looks untriaged on every sweep and
        # is re-classified - and re-charged - every four hours, silently.
        try:
            jira.set_property(args.scratch, PROPERTY_KEY + "-preflight",
                              {"probe": "triage pilot preflight"})
            stored = jira.get_property(args.scratch, PROPERTY_KEY + "-preflight")
            ok &= check("bot can read and write entity properties",
                        (stored or {}).get("probe") == "triage pilot preflight",
                        "idempotency depends on this")
        except Exception as e:
            ok &= check("bot can read and write entity properties", False, str(e)[:200])
        finally:
            try:
                jira.delete_property(args.scratch, PROPERTY_KEY + "-preflight")
            except Exception as e:
                print(f"       WARN: could not delete the preflight property from "
                      f"{args.scratch} ({type(e).__name__}); remove manually")

    elif args.scratch:
        # Asked for the write probes and could not run them. Skipping quietly
        # here let the go-live gate exit 0 having tested nothing - the same
        # false PASS the slash probe used to give, one level up.
        ok &= check("write probes requested but not run", False,
                    "--scratch needs JIRA_EMAIL and JIRA_API_TOKEN; none are set, "
                    "so no write permission was verified")
    else:
        print("       (label-charset write test skipped: pass --scratch O3-XXXX with bot credentials)")

    ok &= check_anthropic(cfg)
    ok &= check_pilot_launch(cfg)
    return 0 if ok else 1


def check_pilot_launch(cfg: dict) -> bool:
    """Is the SLA anchor a date that could have happened?

    Checked here as well as in metrics.py because of when each one runs. This
    value is typed by hand once, at launch; metrics.py first reads it a week
    later, by which point the sweeps have already been measured against it. The
    gate runs before every sweep, so it is where a typo gets caught while it
    still costs nothing.

    Unset stays informational - the sweep does not need it and the pilot has
    run for weeks without it - but a value that is set and impossible fails,
    the same split as the Anthropic credential above.
    """
    value = cfg["metrics"].get("pilot_launch")
    if not value:
        print("       [metrics].pilot_launch: unset - set it at launch; "
              "`python -m triage.metrics` needs it to anchor the 24h SLA")
        return True
    try:
        from .metrics import parse_launch

        parse_launch(value)
    except SystemExit as e:
        return check("[metrics].pilot_launch is a date in the past", False, str(e)[:200])
    return check("[metrics].pilot_launch is a date in the past", True, value)


def check_anthropic(cfg: dict) -> bool:
    """Does the Anthropic credential actually work, and is the pinned model there?

    This line used to report whether an environment variable existed, which is
    not the same claim: a rotated or revoked key sets it just as well as a
    working one. The gate would pass, the sweep would then fail every ticket
    until the breaker aborted it, and the whole reason preflight became a
    workflow step was to spend one runner-minute here instead.

    GET /v1/models is not a generation call, so this bills nothing - measured
    at under a second - and a bad key comes back as a clean 401 rather than as
    31 failed tickets.

    The pinned model is checked against the same response because config.toml
    naming a model this account cannot reach fails identically and just as
    late. Absent credentials stay informational rather than fatal: a dry run
    and a gather need none, and only --live does.
    """
    label = "Anthropic credential works"
    has_env_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
    pinned = cfg["claude"]["model"]
    try:
        available = [m.id for m in Anthropic().models.list(limit=100).data]
    except Exception as e:
        if not has_env_key and "authentication" in f"{type(e).__name__}{e}".lower():
            print("       Anthropic credentials: none configured - fine for a dry run "
                  "or --no-classify; --live needs ANTHROPIC_API_KEY or `ant auth login`")
            return True
        return check(label, False, f"{type(e).__name__}: {e}"[:200])
    source = "ANTHROPIC_API_KEY" if has_env_key else "`ant auth login` profile"
    ok = check(label, True, f"authenticated via {source}")
    # Listed models are paged; only conclude "missing" from a page that could
    # have held it, or a long account listing would fail a model that is there.
    if pinned not in available and len(available) < 100:
        ok &= check(f"pinned model {pinned} is available", False,
                    f"config.toml pins it but this account lists {sorted(available)[:6]}")
    else:
        ok &= check(f"pinned model {pinned} is available", True, "")
    return ok


if __name__ == "__main__":
    sys.exit(main())
