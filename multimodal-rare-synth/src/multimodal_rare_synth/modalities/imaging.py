from __future__ import annotations
from .interfaces import BaseGenerator


class ImagingGenerator(BaseGenerator):
    def generate(self, context: dict) -> dict:
        n = context["n_samples"]
        return {"imaging": {"n_images": n, "model": "stylegan_stub", "status": "stub"}}
