from __future__ import annotations
from .interfaces import BaseGenerator


class OntologyGenerator(BaseGenerator):
    def generate(self, context: dict) -> dict:
        return {
            "ontology": {
                "method": "onto_cgan_stub",
                "supports_unseen_disease": True,
                "status": "stub",
            }
        }
