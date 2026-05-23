"""
FastAPI backend for the Whisper Spanish ASR demo.

Run with:
  uvicorn app.main:app --reload
"""

import os
import re
import tempfile
import torch
import numpy as np
import librosa
import jiwer
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import FileResponse
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# Model loading
# This runs once when the server starts, not on every request.
# Loading a 244M model takes a few seconds, you pay that cost once.

MODEL_PATH = Path("models/whisper-spanish-finetuned")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading model from {MODEL_PATH} on {DEVICE} ...")
processor = WhisperProcessor.from_pretrained(str(MODEL_PATH), language="spanish", task="transcribe")
model = WhisperForConditionalGeneration.from_pretrained(str(MODEL_PATH))
model.to(DEVICE)
model.eval()
print("Model ready.")

# FastAPI app
# `app` is the FastAPI application object. You attach endpoints to it using
# decorators like @app.get(...) and @app.post(...).

app = FastAPI(title="Whisper Spanish ASR")


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return " ".join(text.split())


# Endpoint 1: GET /
# When the browser navigates to http://localhost:8000/, it sends a GET request.
# This endpoint responds by sending back the index.html file.
# That's it, the browser renders the HTML and the UI appears.

@app.get("/")
async def root():
    return FileResponse("app/static/index.html")


# Endpoint 2: POST /transcribe
# When the user clicks "Transcribe", the browser sends a POST request here.
# The request body contains:
#   - audio: the audio file as binary data (UploadFile)
#   - reference: an optional text string (Form field)
#
# FastAPI reads these from the request automatically and passes them as
# arguments to this function. You don't write any parsing code.
#
# The function returns a dict. FastAPI automatically converts it to JSON
# and sends it back to the browser.

@app.post("/transcribe")
async def transcribe(
    audio: UploadFile = File(...),
    reference: str = Form(""),
):
    audio_bytes = await audio.read()
    suffix = Path(audio.filename).suffix if audio.filename else ".webm"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        # Load audio, resample to 16 kHz mono (what Whisper expects)
        audio_array, _ = librosa.load(tmp_path, sr=16000, mono=True, dtype=np.float32)

        # Run the model
        inputs = processor(audio_array, sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
            predicted_ids = model.generate(
                inputs.input_features.to(DEVICE),
                language="spanish",
                task="transcribe",
            )
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

        # Build response dict, FastAPI turns this into JSON automatically
        result = {"transcription": transcription, "wer": None, "cer": None}

        if reference.strip():
            wer = jiwer.wer(normalize(reference), normalize(transcription))
            cer = jiwer.cer(normalize(reference), normalize(transcription))
            result["wer"] = round(wer * 100, 1)
            result["cer"] = round(cer * 100, 1)

        return result

    except Exception as exc:
        # Return the error as JSON so the browser can display it
        return {"error": str(exc), "transcription": "", "wer": None, "cer": None}

    finally:
        os.unlink(tmp_path)
