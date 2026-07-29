"""
FastAPI wrapper around the existing SpecTUS inference pipeline (predict.py).

Loads the model once at process startup and serves single-spectrum predictions
over HTTP, reusing the same datapipe/collator/generation code path as predict.py
so behavior matches the validated CLI exactly.
"""

import json
import os
import tempfile
from copy import deepcopy
from typing import Dict, List

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from model.modeling_spectus import SpectusForConditionalGeneration
from utils.data_utils import SpectroDataCollator, build_single_datapipe
from utils.general_utils import build_tokenizer, get_sequence_probs
from predict import get_unique_predictions, get_canon_predictions, prepare_decoder_input

CHECKPOINT = os.environ.get("SPECTUS_CHECKPOINT", "/app/checkpoint")
TOKENIZER_PATH = os.environ.get("SPECTUS_TOKENIZER", "/app/tokenizer/tokenizer_mf10M.model")
DEVICE = os.environ.get("DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
SOURCE_TOKEN = "<nist>"

# Same constants as configs/predict_own.yaml - properties of the trained model, not
# request-configurable.
BASE_PREPROCESS_ARGS = {
    "source_token": SOURCE_TOKEN,
    "restrict_intensities": False,
    "inference_mode": True,
    "max_num_peaks": 300,
    "max_mol_repr_len": 100,
    "max_mz": 500,
    "mol_repr": "smiles",
    "log_base": 1.28,
    "log_shift": 29,
    "max_cumsum": None,
}
LENGTH_PENALTY = 1.0

tokenizer = build_tokenizer(TOKENIZER_PATH)
model = SpectusForConditionalGeneration.from_pretrained(CHECKPOINT)
model.generation_config.length_penalty = LENGTH_PENALTY
model.eval()
model.to(DEVICE)

app = FastAPI(title="SpecTUS")


class PredictRequest(BaseModel):
    mz: List[float]
    intensity: List[float]
    num_beams: int = 10
    num_return_sequences: int = 10


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(req: PredictRequest) -> Dict[str, Dict[str, float]]:
    if len(req.mz) != len(req.intensity):
        raise HTTPException(status_code=422, detail="mz and intensity must be the same length")

    generation_config = {
        "top_k": None,
        "top_p": None,
        "do_sample": False,
        "num_beams": req.num_beams,
        "temperature": None,
        "penalty_alpha": None,
        "num_return_sequences": req.num_return_sequences,
        "max_length": 200,
        "length_penalty": LENGTH_PENALTY,
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        # "smiles" is required by filter_datapoints as a validity gate, but is never
        # used for generation under inference_mode - the real molecule is unknown.
        json.dump({"mz": req.mz, "intensity": req.intensity, "smiles": "C"}, f)
        f.write("\n")
        temp_path = f.name

    try:
        preprocess_args = deepcopy(BASE_PREPROCESS_ARGS)
        preprocess_args["tokenizer"] = tokenizer
        datapipe = build_single_datapipe(
            temp_path,
            shuffle=False,
            source_token=preprocess_args.pop("source_token"),
            preprocess_args=preprocess_args,
        )
        loader = torch.utils.data.DataLoader(
            datapipe,
            batch_size=1,
            collate_fn=SpectroDataCollator(inference_mode=True, keep_all_columns=True),
        )
        batch = next(iter(loader), None)
        if batch is None:
            raise HTTPException(status_code=422, detail="Spectrum was filtered out (too many peaks or m/z out of range)")
    finally:
        os.remove(temp_path)

    decoder_input_ids = prepare_decoder_input(SOURCE_TOKEN, tokenizer, batch_size=1)

    model_input = {key: value.to(DEVICE) for key, value in batch.items() if key in ["input_ids", "position_ids"]}
    with torch.no_grad():
        generated_outputs = model.generate(
            decoder_input_ids=decoder_input_ids.to(DEVICE),
            **model_input,
            **generation_config,
            output_scores=True,
            return_dict_in_generate=True,
        )

    preds = tokenizer.batch_decode(generated_outputs.sequences.tolist(), skip_special_tokens=True)
    preds = [preds]  # single-item batch

    unique_preds, unique_idxs = get_unique_predictions(preds)
    canon_preds, canon_idxs = get_canon_predictions(unique_preds, unique_idxs)
    all_probs = get_sequence_probs(model, generated_outputs, 1, generation_config["num_beams"] > 1)

    predictions = {
        canon_preds[0][j]: all_probs[0, canon_idx].item()
        for j, canon_idx in enumerate(canon_idxs[0])
    }
    return {"predictions": predictions}
