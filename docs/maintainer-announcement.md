# AI triage pilot on O3 — starting [DATE], and three things we need from you

We're running a time-boxed pilot to help sort the O3 backlog. A bot account
(**[BOT DISPLAY NAME]**) will add **one** `ai-triage-*` label and one
explanatory comment for it to O3 tickets that are in **To Do**, created since
**[COHORT START DATE]**, and have no linked pull request. That's roughly 30
tickets to begin with, then new ones as they arrive.

If a ticket is edited enough to change the answer it is re-classified: the old
label comes off, the new one goes on, and one comment explains the new label. So
a ticket can end up with two comments over the pilot, never with two labels.

The goal is narrow: make it easier to see which tickets are ready to pick up,
which are waiting on a decision, and which can't be started yet because we don't
know what "done" means.

## The three labels

| Label | Means |
|---|---|
| `ai-triage-automation-candidate` | The ticket says what needs to happen and there's a clear way to check it's done. |
| `ai-triage-needs-judgment` | Doing it right depends on a call the ticket can't settle — clinical meaning, product intent, or how something should look. |
| `ai-triage-needs-more-info` | Even after reading the parent ticket and the comments, we still can't tell what "done" looks like. The comment lists what's missing. |

**The label never records an opinion about the work itself.** It only records
whether a human judgement is needed — not what that judgement should be. The bot
makes no technical or clinical calls.

It also only reads what's already on the ticket: summary, description, acceptance
criteria, the parent ticket, linked ticket titles, and comments from people. It
never changes status, assignee, description, links, or anything else. Labels and
comments are the only things it writes that you can see; it also stores a small
record of what it decided against the issue, invisible in the UI, which is how
it knows not to label the same ticket twice.

It will sometimes be wrong. Which brings us to the first ask.

## 1. If a label is wrong, remove it

Removing an `ai-triage-*` label is how you tell us the label was wrong, and it's
also a **permanent opt-out** — that ticket will not be labelled again for the
rest of the pilot. No need to explain, though we'd love to hear why.

**Putting the label back does not undo it.** The opt-out is read from the
ticket's history, so a removal counts even if you restore the label a minute
later; the ticket stays out for good, and the restore also shows up in our
weekly digest as a manual label change. If you remove one by accident, tell us
in the thread rather than trying to repair it on the ticket.

Two things follow from that:

- **Please don't remove the label just to tidy up.** It's our main quality
  signal: more than 1 in 10 removed and that measure fails, costing the pilot its
  strongest result; past 1 in 5 we stop outright. Removals
  meant as housekeeping would read as "the triage is bad."
- **Leave it in place after promoting a ticket to `intro`.** A ticket can hold
  both, and it needs to hold both to count for us — see ask 3.

## 2. Put the O3 key in your PR title

The pilot skips tickets that already have a pull request, so nobody gets triage
noise on work that's already underway. It finds those PRs two ways: Jira's
development panel, and by searching GitHub for the ticket key.

Both depend on the key appearing in the PR. So please use the form the majority
already do:

```
(fix) O3-1234: short description
```

This isn't hypothetical. One ticket in the starting cohort has an open PR that
the pilot can't connect to it, because the PR title doesn't mention the key —
so that ticket will get triaged as though nobody were working on it. Keys in PR
titles also fix the dev panel, the release notes, and anyone searching Jira for
"what happened to this."

## 3. Apply `intro` or `not-intro` when you act on a triaged ticket

If you look at a triaged ticket and decide it's a good first issue for a
newcomer, add `intro`. If you decide it isn't, add `not-intro`. Either is useful;
silence isn't.

Being straight with you about why: "did triage help us produce
newcomer-ready tickets" is one of three pre-registered measures this pilot is
judged on, and it currently sits at **zero** — no ticket created in the cohort
window carries `intro`, and `not-intro` has never been used on O3 at all. If
this habit doesn't take hold, the pilot cannot show it succeeded even if the
triage itself is good.

## What happens next

Weekly, we publish three numbers — how quickly tickets got sorted, what fraction
of labels were removed, and how many triaged tickets became `intro`/`not-intro`
outcomes — against thresholds we committed to before starting, and a resulting
**adopt / extend / stop** call. The decision rule was fixed in advance
specifically so we can't move the goalposts after seeing results.

Pilot runs for **[DURATION]**. Questions, complaints, and "this label is
nonsense" all welcome at **[CHANNEL / THREAD]**.

— [DENNIS] & [VERONICA]
