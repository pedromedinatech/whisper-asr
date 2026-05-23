"""
Reads recordings from data/personal/audio/ and their metadata from
data/personal/metadata.csv, extracts Whisper features, and saves the
resulting HuggingFace Dataset to data/personal/processed/.

The processed dataset can then be merged into the training set for Stage 4.

metadata.csv columns:
  filename, duration_s, difficulty, transcript, notes
"""

import re
import torch
import librosa
import numpy as np
import pandas as pd
from pathlib import Path
from datasets import Dataset
from transformers import WhisperProcessor

MODEL_ID = "openai/whisper-small"
AUDIO_DIR = Path("data/personal/audio")
META_PATH = Path("data/personal/metadata.csv")
OUT_DIR = Path("data/personal/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MIN_DURATION_S = 0.5
MAX_DURATION_S = 29.5


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = " ".join(text.split())
    return text


def load_audio(path: Path) -> tuple[np.ndarray, int]:
    audio, sr = librosa.load(str(path), sr=16000, mono=True, dtype=np.float32)
    return audio, sr


print(f"Loading processor from {MODEL_ID} ...")
processor = WhisperProcessor.from_pretrained(MODEL_ID, language="spanish", task="transcribe")

meta = pd.read_csv(META_PATH)
print(f"Found {len(meta)} entries in metadata.csv")

records = []
skipped = 0

for _, row in meta.iterrows():
    filename = str(row["filename"])
    transcript = str(row.get("transcript", ""))
    difficulty = int(row.get("difficulty", 1))

    audio_path = AUDIO_DIR / filename
    if not audio_path.exists():
        print(f"MISSING: {audio_path}")
        skipped += 1
        continue

    audio, _ = load_audio(audio_path)

    duration = len(audio) / 16000.0
    if not (MIN_DURATION_S <= duration <= MAX_DURATION_S):
        print(f"SKIP (duration {duration:.1f}s out of range): {filename}")
        skipped += 1
        continue

    norm = normalize(transcript)
    if not norm:
        print(f"SKIP (empty transcript): {filename}")
        skipped += 1
        continue

    inputs = processor(audio, sampling_rate=16000, return_tensors="np")
    labels = processor.tokenizer(norm).input_ids

    records.append({
        "input_features": inputs.input_features[0],
        "labels": labels,
        "filename": filename,
        "difficulty": difficulty,
        "transcript": norm,
        "duration_s": round(duration, 2),
    })

print(f"\nProcessed: {len(records)} valid samples, {skipped} skipped")

if records:
    ds = Dataset.from_list(records)
    ds.save_to_disk(str(OUT_DIR))
    print(f"Saved to {OUT_DIR}")

    total_minutes = sum(r["duration_s"] for r in records) / 60
    print(f"Total audio: {total_minutes:.1f} min")

    diff_counts = pd.Series([r["difficulty"] for r in records]).value_counts().sort_index()
    for level, count in diff_counts.items():
        print(f"Difficulty {level}: {count} samples")

    sample_df = pd.DataFrame([{
        "filename": r["filename"],
        "duration_s": r["duration_s"],
        "difficulty": r["difficulty"],
        "transcript": r["transcript"],
    } for r in records])

    sample_path = Path("results/benchmark/03_personal_sample.csv")
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    sample_df.to_csv(sample_path, index=False)
    print(f"Sample preview saved to {sample_path}")
else:
    print("No valid samples, check audio files and metadata.csv")

print("ok")
