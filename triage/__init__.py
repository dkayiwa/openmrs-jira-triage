"""O3 AI triage pilot: a deterministic pipeline with one Claude call per ticket.

The model only returns a classification object; every Jira read and write
happens in this code, so the pilot's guarantees (visible information only,
labels + comments only, opt-out on label removal) are properties of the code
and the bot's permission scheme, not of the prompt.
"""
