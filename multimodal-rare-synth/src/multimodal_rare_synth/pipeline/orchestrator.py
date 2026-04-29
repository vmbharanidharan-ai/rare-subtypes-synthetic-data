from __future__ import annotations

from multimodal_rare_synth.config import AppConfig
from multimodal_rare_synth.modalities.imaging import ImagingGenerator
from multimodal_rare_synth.modalities.genomics import GenomicsGenerator
from multimodal_rare_synth.modalities.clinical import ClinicalGenerator
from multimodal_rare_synth.modalities.multiomics import MultiOmicsIntegrator
from multimodal_rare_synth.modalities.ontology import OntologyGenerator
from multimodal_rare_synth.validation.evaluator import evaluate


class MultimodalOrchestrator:
    def __init__(self, config: AppConfig):
        self.config = config

    def run(self) -> dict:
        context = {
            "n_samples": self.config.generation.n_samples,
            "seed": self.config.generation.seed,
            "subtype": self.config.project.subtype,
        }
        out: dict = {"project": self.config.project.__dict__, "generation": self.config.generation.__dict__}

        if self.config.modalities.imaging:
            out.update(ImagingGenerator().generate(context))
        if self.config.modalities.genomics:
            out.update(GenomicsGenerator().generate(context))
        if self.config.modalities.clinical:
            out.update(ClinicalGenerator().generate(context))
        if self.config.modalities.multiomics:
            out.update(MultiOmicsIntegrator().generate(context))
        if self.config.modalities.ontology:
            out.update(OntologyGenerator().generate(context))

        out.update(
            evaluate(
                out,
                {
                    "fidelity": self.config.validation.enable_fidelity,
                    "utility": self.config.validation.enable_utility,
                    "privacy": self.config.validation.enable_privacy,
                    "bio_consistency": self.config.validation.enable_bio_consistency,
                },
            )
        )
        return out
