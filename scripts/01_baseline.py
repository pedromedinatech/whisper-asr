"""
Zero-shot baseline
Measures WER and CER of Whisper-small on Spanish with no fine-tuning.
This is the reference point against which the fine-tuned model will be compared.
"""

import io
import re
import torch
import numpy as np
import torchaudio
import soundfile as sf
import jiwer
import pandas as pd
from pathlib import Path
from datasets import load_dataset, Audio
from transformers import WhisperProcessor, WhisperForConditionalGeneration


MODEL_ID = "openai/whisper-small"
DATASET_ID = "facebook/voxpopuli"
LANGUAGE_CODE = "es"
MAX_SAMPLES = 500   # increase for a more stable WER estimate
RESULTS_DIR = Path("results/benchmark")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

processor = WhisperProcessor.from_pretrained(
    MODEL_ID, language="spanish", task="transcribe"
)

model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID)
model.to(device)
model.eval()
print("Model ok.")

# Only the test split is used here.

print(f"Streaming {DATASET_ID} ({LANGUAGE_CODE}) test split ...")
dataset = load_dataset(
    DATASET_ID,
    LANGUAGE_CODE,
    split="test",
    streaming=True,
)

# Disable automatic audio decoding
dataset = dataset.cast_column("audio", Audio(decode=False))

# Text normalization
# Applied identically to both the reference transcript and the model output
# so that differences in capitalisation or punctuation don't inflate WER.

def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)   # remove punctuation
    text = " ".join(text.split())   # collapse whitespace
    return text


# Inference loop

references = []
hypotheses = []

print(f"Running inference on {MAX_SAMPLES} samples ...")
for i, sample in enumerate(dataset):
    if i >= MAX_SAMPLES:
        break

    # decode=False gives us raw bytes and we decode with soundfile to avoid torchcodec
    raw = sample["audio"]
    audio_bytes = raw["bytes"] if raw["bytes"] else open(raw["path"], "rb").read()
    audio, src_sr = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=False)

    # Whisper requires 16 kHz mono
    if src_sr != 16000:
        audio_tensor = torch.from_numpy(audio).unsqueeze(0)
        audio_tensor = torchaudio.functional.resample(audio_tensor, src_sr, 16000)
        audio = audio_tensor.squeeze(0).numpy()

    # Audio to log-mel spectrogram (80 bins × 3000 frames for 30 s)
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
    input_features = inputs.input_features.to(device)

    # Autoregressive decoding with beam search
    with torch.no_grad():
        predicted_ids = model.generate(
            input_features,
            language="spanish",
            task="transcribe",
        )

    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

    references.append(normalize(sample["normalized_text"]))
    hypotheses.append(normalize(transcription))

    if (i + 1) % 50 == 0:
        partial_wer = jiwer.wer(references, hypotheses)
        print(f"[{i + 1:>4}/{MAX_SAMPLES}] partial WER: {partial_wer * 100:.2f}%")


# Metrics

wer = jiwer.wer(references, hypotheses)
cer = jiwer.cer(references, hypotheses)

print()
print(f"Model: {MODEL_ID} (zero-shot, no fine-tuning)")
print(f"Dataset: VoxPopuli / es / test")
print(f"Samples: {len(references)}")
print(f"WER: {wer * 100:.2f}%")
print(f"CER: {cer * 100:.2f}%")

# Save to CSV

results = pd.DataFrame([{
    "model": MODEL_ID,
    "variant": "zero-shot",
    "dataset": f"voxpopuli/{LANGUAGE_CODE}/test",
    "samples": len(references),
    "wer": round(wer, 4),
    "cer": round(cer, 4),
}])

output_path = RESULTS_DIR / "01_baseline.csv"
results.to_csv(output_path, index=False)
print(f"Saved to {output_path}")
