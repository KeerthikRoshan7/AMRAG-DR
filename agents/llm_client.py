"""
Shared Groq client for all agentic reasoning components. Uses
llama-3.3-70b-versatile, matching the EMRA-DR zero-cost deployment stack.
"""

import os
import json
from groq import Groq

MODEL_NAME = "llama-3.3-70b-versatile"


class LLMClient:
    def __init__(self, api_key: str | None = None):
        api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY not set. Get a free key at https://console.groq.com "
                "and put it in your .env file."
            )
        self.client = Groq(api_key=api_key)

    def complete(self, system_prompt: str, user_prompt: str,
                 temperature: float = 0.2, json_mode: bool = False) -> str:
        kwargs = {}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        response = self.client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            **kwargs,
        )
        return response.choices[0].message.content

    def complete_json(self, system_prompt: str, user_prompt: str, temperature: float = 0.1) -> dict:
        raw = self.complete(system_prompt, user_prompt, temperature=temperature, json_mode=True)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            # Groq occasionally wraps JSON in prose despite json_mode; strip fences as a fallback.
            cleaned = raw.strip().strip("```json").strip("```").strip()
            return json.loads(cleaned)
