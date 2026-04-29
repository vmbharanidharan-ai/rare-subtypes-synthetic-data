from __future__ import annotations
from .interfaces import BaseGenerator


class GenomicsGenerator(BaseGenerator):
    def generate(self, context: dict) -> dict:
        n = context["n_samples"]
        return {"genomics": {"n_profiles": n, "model": "vae_gan_stub", "status": "stub"}}
