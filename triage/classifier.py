"""One Claude call per ticket, returning a schema-validated classification.

The model never touches Jira: it receives pre-assembled visible-information
text and returns JSON. A prompt-injected ticket can therefore at worst
mislabel itself, which human review (and the removal metric) then catches.

Server-side refusal fallbacks are enabled so a stray safety-classifier decline
re-runs on Anthropic's default fallback model instead of wedging a run.
"""
from __future__ import annotations

import json
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


def validate_classification(data: dict) -> list[str]:
    """Schema violations in one classification object; empty when valid.

    The API path has this enforced server-side by output_config. Anything
    produced elsewhere - an agent in a Claude Code session, a hand-written file -
    is unvalidated input on its way to a public Jira ticket, so it is checked
    here before it can be applied.
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
        elif spec["type"] == "number" and not isinstance(value, (int, float)):
            errors.append(f"{key} must be a number")
        elif spec["type"] == "array" and (
            not isinstance(value, list) or any(not isinstance(x, str) for x in value)
        ):
            errors.append(f"{key} must be an array of strings")
        if "enum" in spec and value not in spec["enum"]:
            errors.append(f"{key} must be one of {spec['enum']}")
    return errors


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
        return Classification(
            label=data["label"],
            rationale=data["rationale"],
            missing_info=list(data.get("missing_info") or []),
            verification_steps=list(data.get("verification_steps") or []),
            confidence=float(data["confidence"]),
            model=response.model,
        )
