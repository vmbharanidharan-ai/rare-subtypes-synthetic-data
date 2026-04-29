from __future__ import annotations
from .interfaces import BaseGenerator


class ClinicalGenerator(BaseGenerator):
    def generate(self, context: dict) -> dict:
        n = context["n_samples"]
        return {"clinical": {"n_rows": n, "model": "ctgan_stub", "status": "stub"}}
