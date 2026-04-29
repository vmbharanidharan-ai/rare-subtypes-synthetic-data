from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import yaml


@dataclass
class ProjectConfig:
    name: str
    disease_focus: str
    subtype: str


@dataclass
class ModalitiesConfig:
    imaging: bool
    genomics: bool
    clinical: bool
    multiomics: bool
    ontology: bool


@dataclass
class GenerationConfig:
    n_samples: int
    seed: int


@dataclass
class ValidationConfig:
    enable_fidelity: bool
    enable_utility: bool
    enable_privacy: bool
    enable_bio_consistency: bool


@dataclass
class AppConfig:
    project: ProjectConfig
    modalities: ModalitiesConfig
    generation: GenerationConfig
    validation: ValidationConfig


def load_config(path: str | Path) -> AppConfig:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    return AppConfig(
        project=ProjectConfig(**raw["project"]),
        modalities=ModalitiesConfig(**raw["modalities"]),
        generation=GenerationConfig(**raw["generation"]),
        validation=ValidationConfig(**raw["validation"]),
    )
