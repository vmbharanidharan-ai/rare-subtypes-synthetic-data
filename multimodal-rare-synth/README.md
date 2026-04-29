# multimodal-rare-synth (next-gen scaffold)

A second project scaffold for **rare oncology synthetic data generation** across five modalities:
1. imaging (histopathology / MRI / CT)
2. genomic / multi-omics
3. clinical tabular / longitudinal
4. cross-modal integration
5. ontology-informed generation for ultra-rare subtypes

## What this scaffold gives you

- modular interfaces for each modality-specific generator
- one orchestrator to run the full multimodal stack
- validation hooks for fidelity / utility / privacy / biological consistency
- API skeleton to trigger generation jobs

## Architecture

```text
Data Sources -> Harmonization -> Modality Generators -> Cross-Modal Consistency -> Validation -> Artifacts/API

Modalities:
  ImagingGenerator
  GenomicsGenerator
  ClinicalGenerator
  MultiOmicsIntegrator
  OntologyGenerator
```

## Quick start

```bash
cd multimodal-rare-synth
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m multimodal_rare_synth.api.server
```

Then open `http://127.0.0.1:8010/docs`.

## Next implementation priorities

1. plug in a real imaging model (StyleGAN/latent diffusion)
2. plug in omics model (VAE/Transformer + condition vectors)
3. add graph/ontology encoder for ultra-rare transfer
4. implement cross-modal consistency losses + calibration
5. add clinical expert review pipeline and model cards
