from __future__ import annotations

from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

from multimodal_rare_synth.config import load_config
from multimodal_rare_synth.pipeline.orchestrator import MultimodalOrchestrator

app = FastAPI(title="Multimodal Rare Synth API", version="0.1.0")


class GenerateRequest(BaseModel):
    config_path: str = "configs/default.yaml"


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/generate")
def generate(req: GenerateRequest) -> dict:
    cfg = load_config(req.config_path)
    result = MultimodalOrchestrator(cfg).run()
    return result


if __name__ == "__main__":
    uvicorn.run("multimodal_rare_synth.api.server:app", host="127.0.0.1", port=8010, reload=False)
