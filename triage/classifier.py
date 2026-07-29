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
        text = next(b.text for b in response.content if b.type == "text")
        data = json.loads(text)
        return Classification(
            label=data["label"],
            rationale=data["rationale"],
            missing_info=list(data.get("missing_info") or []),
            verification_steps=list(data.get("verification_steps") or []),
            confidence=float(data["confidence"]),
            model=response.model,
        )
