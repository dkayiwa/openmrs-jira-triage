"""One Claude call per ticket, returning a schema-validated classification.

The model never touches Jira: it receives pre-assembled visible-information
text and returns JSON. A prompt-injected ticket can therefore at worst
mislabel itself, which human review (and the removal metric) then catches.

Server-side refusal fallbacks are enabled so a stray safety-classifier decline
re-runs on Anthropic's default fallback model instead of wedging a run.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass

from anthropic import Anthropic

LABEL_KEYS = ["automation_candidate", "needs_judgment", "needs_more_info"]

SCHEMA = {
    "type": "object",
    "properties": {
        "label": {"type": "string", "enum": LABEL_KEYS},
        "rationale": {"type": "string"},
        "missing_info": {"type": "array", "items": {"type": "string"}},
        "verification_steps": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "number"},
    },
    "required": ["label", "rationale", "missing_info", "verification_steps", "confidence"],
    "additionalProperties": False,
}


# Sanity bounds on text that will be posted to a public ticket. The prompt asks
# the model for at most two sentences, but a classifications file is not bound by
# the prompt, and structured-output schemas cannot express these limits (numeric
# and length constraints are rejected by the API), so they are enforced here for
# both paths.
MAX_RATIONALE = 2000
MAX_ITEM = 300
MAX_ITEMS = 20


def validate_classification(data: dict) -> list[str]:
    """Schema and sanity violations in one classification object; empty if valid.

    Called on both paths. output_config constrains the API response's *shape*
    server-side but not its magnitudes, and a classifications file is
    unvalidated input on its way to a public Jira ticket.
    """
    errors = []
    props = SCHEMA["properties"]
    for key in SCHEMA["required"]:
        if key not in data:
            errors.append(f"missing {key!r}")
    for key, value in data.items():
        spec = props.get(key)
        if spec is None:
            errors.append(f"unexpected field {key!r}")
            continue
        if spec["type"] == "string" and not isinstance(value, str):
            errors.append(f"{key} must be a string")
        # bool is an int in Python, so `confidence: true` would otherwise pass
        # and render as a confidence of 1.00.
        elif spec["type"] == "number" and (isinstance(value, bool)
                                           or not isinstance(value, (int, float))):
            errors.append(f"{key} must be a number")
        elif spec["type"] == "array" and (
            not isinstance(value, list) or any(not isinstance(x, str) for x in value)
        ):
            errors.append(f"{key} must be an array of strings")
        if "enum" in spec and value not in spec["enum"]:
            errors.append(f"{key} must be one of {spec['enum']}")
    # Magnitudes, which the schema cannot carry.
    confidence = data.get("confidence")
    if (isinstance(confidence, (int, float)) and not isinstance(confidence, bool)
            and not 0.0 <= float(confidence) <= 1.0):
        errors.append(f"confidence must be between 0 and 1, got {confidence}")
    rationale = data.get("rationale")
    if isinstance(rationale, str) and not rationale.strip():
        # The comment exists to explain the label; a blank one is worse than none.
        errors.append("rationale is empty")
    if isinstance(rationale, str) and len(rationale) > MAX_RATIONALE:
        errors.append(f"rationale is {len(rationale)} chars, over the {MAX_RATIONALE} limit")
    for key in ("missing_info", "verification_steps"):
        items = data.get(key)
        if not isinstance(items, list):
            continue
        if len(items) > MAX_ITEMS:
            errors.append(f"{key} has {len(items)} items, over the {MAX_ITEMS} limit")
        for item in items:
            if isinstance(item, str) and len(item) > MAX_ITEM:
                errors.append(f"{key} item is {len(item)} chars, over the {MAX_ITEM} limit")
    return errors


def clamp_classification(data: dict) -> list[str]:
    """Bound magnitudes in place; return notes on what was adjusted.

    Used on the API path, where rejecting an otherwise-correct classification is
    the expensive choice: the ticket writes no entity property, so its content
    hash is unchanged and it is re-classified - and re-charged - on every sweep,
    and five such tickets in a row abort the whole sweep. Since scope_jql orders
    by created ASC, that starves every newer ticket, which is the population the
    24h SLA measures. A slightly-long comment is a far smaller problem.

    The file path deliberately rejects instead, because its author can fix it.
    """
    notes = []
    confidence = data.get("confidence")
    if isinstance(confidence, bool):
        data["confidence"] = float(confidence)
        notes.append("confidence was a boolean")
        confidence = data["confidence"]
    if isinstance(confidence, (int, float)) and not 0.0 <= float(confidence) <= 1.0:
        data["confidence"] = min(1.0, max(0.0, float(confidence)))
        notes.append(f"confidence {confidence} clamped to {data['confidence']}")
    rationale = data.get("rationale")
    if isinstance(rationale, str):
        # Emptiness is tested before truncation, so whitespace-only text of any
        # length lands here rather than being truncated to a bare ellipsis.
        # Substituted rather than rejected for the reason in the docstring: a
        # blank rationale is schema-valid (SCHEMA cannot express minLength), so
        # rejecting it would re-charge this ticket on every sweep forever. The
        # marker keeps the label and says plainly that no explanation came back.
        if not rationale.strip():
            data["rationale"] = "(the classifier returned no rationale for this label)"
            notes.append("rationale was empty")
        elif len(rationale) > MAX_RATIONALE:
            data["rationale"] = rationale[: MAX_RATIONALE - 1].rstrip() + "…"
            notes.append(f"rationale truncated from {len(rationale)} chars")
    for key in ("missing_info", "verification_steps"):
        items = data.get(key)
        if not isinstance(items, list):
            continue
        if len(items) > MAX_ITEMS:
            items = items[:MAX_ITEMS]
            notes.append(f"{key} truncated to {MAX_ITEMS} items")
        data[key] = [
            item[: MAX_ITEM - 1].rstrip() + "…"
            if isinstance(item, str) and len(item) > MAX_ITEM else item
            for item in items
        ]
        if any(isinstance(i, str) and len(i) > MAX_ITEM for i in items):
            notes.append(f"{key} item(s) truncated to {MAX_ITEM} chars")
    return notes


@dataclass
class Classification:
    label: str
    rationale: str
    missing_info: list[str]
    verification_steps: list[str]
    confidence: float
    model: str
    refused: bool = False


class Classifier:
    def __init__(self, model: str, max_tokens: int, system_prompt: str):
        # Zero-arg client: uses ANTHROPIC_API_KEY or an `ant auth login` profile.
        self.client = Anthropic()
        self.model = model
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt

    def classify(self, context: str) -> Classification:
        response = self.client.beta.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            betas=["server-side-fallback-2026-07-01"],
            system=self.system_prompt,
            messages=[{"role": "user", "content": context}],
            # extra_body keeps this working on SDK versions that don't type
            # output_config/fallbacks yet; the wire shape is identical.
            extra_body={
                "output_config": {"format": {"type": "json_schema", "schema": SCHEMA}},
                "fallbacks": "default",
            },
        )
        if response.stop_reason == "refusal":
            return Classification("", "", [], [], 0.0, response.model, refused=True)
        if response.stop_reason == "max_tokens":
            # Thinking is on by default on Opus 5 and shares the max_tokens
            # budget with the response, so a hard ticket can truncate the JSON.
            # Caught here because the bare JSONDecodeError it otherwise causes
            # is undiagnosable, and it would recur on every sweep: the content
            # hash is unchanged, so the ticket never stops being retried.
            raise RuntimeError(
                f"classification truncated at max_tokens={self.max_tokens}; "
                "raise [claude].max_tokens in config.toml"
            )
        block = next((b for b in response.content if b.type == "text"), None)
        if block is None:
            raise RuntimeError(f"no text block in response (stop_reason={response.stop_reason})")
        data = json.loads(block.text)
        # Normalise representation before validating: the model may send null
        # for an empty list, or a numeric string. output_config should prevent
        # both and neither is a semantic problem - magnitudes are what matter,
        # and the schema cannot express those.
        for field in ("missing_info", "verification_steps"):
            if data.get(field) is None:
                data[field] = []
        try:
            data["confidence"] = float(data["confidence"])
        except (KeyError, TypeError, ValueError):
            pass
        notes = clamp_classification(data)
        if notes:
            print(f"WARN: adjusted the model's classification: {'; '.join(notes)}",
                  file=sys.stderr)
        errors = validate_classification(data)
        if errors:
            raise RuntimeError(f"model returned an unusable classification: {'; '.join(errors)}")
        return Classification(
            label=data["label"],
            rationale=data["rationale"],
            missing_info=list(data["missing_info"]),
            verification_steps=list(data["verification_steps"]),
            confidence=float(data["confidence"]),
            model=response.model,
        )
