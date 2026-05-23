"""
Sources:
  VoxPopuli: facebook/voxpopuli, config 'es'
  MLS: facebook/multilingual_librispeech, config 'spanish'

Also runs a zero-shot Whisper benchmark on the MLS test split so that the
result is directly comparable with the VoxPopuli baseline in 01_baseline.py.
"""

import io
import re
import torch
import torchaudio
import soundfile as sf
import numpy as np
import pandas as pd
import jiwer
from pathlib import Path
from datasets import load_dataset, Audio, Dataset, concatenate_datasets
from transformers import WhisperProcessor, WhisperForConditionalGeneration

MODEL_ID = "openai/whisper-small"
VP_DIR = Path("data/opendata/processed/voxpopuli_es")
MLS_DIR = Path("data/opendata/processed/mls_es")
COMBINED_DIR = Path("data/opendata/processed/combined_es")
RESULTS_DIR = Path("results/benchmark")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MLS_BENCHMARK_SAMPLES = 500

SAMPLE_LIMITS = {
    "voxpopuli": {"train": 5000, "validation": 500},
    "mls": {"train": 5000, "validation": 500},
}

MIN_DURATION_S = 0.5
MAX_DURATION_S = 29.5
SHUFFLE_SEED = 42

for d in [VP_DIR, MLS_DIR, COMBINED_DIR]:
    d.mkdir(parents=True, exist_ok=True)

print(f"Loading processor from {MODEL_ID} ...")
processor = WhisperProcessor.from_pretrained(MODEL_ID, language="spanish", task="transcribe")


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = " ".join(text.split())
    return text


def resample_if_needed(audio: np.ndarray, src_sr: int) -> np.ndarray:
    if src_sr == 16000:
        return audio
    tensor = torch.from_numpy(audio).unsqueeze(0)
    tensor = torchaudio.functional.resample(tensor, src_sr, 16000)
    return tensor.squeeze(0).numpy()


def extract_features(audio: np.ndarray, text: str) -> dict | None:
    """Convert 16kHz mono audio + transcript to Whisper input_features and label token IDs."""
    duration = len(audio) / 16000.0
    if not (MIN_DURATION_S <= duration <= MAX_DURATION_S):
        return None
    norm = normalize(text)
    if not norm:
        return None
    inputs = processor(audio, sampling_rate=16000, return_tensors="np")
    labels = processor.tokenizer(norm).input_ids
    return {"input_features": inputs.input_features[0], "labels": labels}


def process_and_save(source: str, split: str, out_dir: Path) -> Dataset:
    """Stream, process, and save one source+split. Returns the Dataset (from disk if already done)."""
    marker = out_dir / split / "dataset_info.json"
    if marker.exists():
        print(f"{source} ({split}): already on disk, loading ...")
        return Dataset.load_from_disk(str(out_dir / split))

    if source == "voxpopuli":
        ds = load_dataset("facebook/voxpopuli", "es", split=split, streaming=True)
        text_col = "normalized_text"
    else:
        mls_split = "dev" if split == "validation" else split
        ds = load_dataset("facebook/multilingual_librispeech", "spanish", split=mls_split, streaming=True)
        text_col = "transcript"

    ds = ds.cast_column("audio", Audio(decode=False))
    ds = ds.take(SAMPLE_LIMITS[source][split])

    print(f"Streaming {source} ({split}, up to {SAMPLE_LIMITS[source][split]}) ...")
    results = []
    for i, sample in enumerate(ds):
        raw = sample["audio"]
        audio_bytes = raw["bytes"] if raw["bytes"] else open(raw["path"], "rb").read()
        audio, src_sr = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=False)
        audio = resample_if_needed(audio, src_sr)
        feat = extract_features(audio, sample.get(text_col, ""))
        if feat:
            results.append(feat)
        if (i + 1) % 500 == 0:
            print(f"{i + 1}/{SAMPLE_LIMITS[source][split]} scanned, {len(results)} valid")

    ds_out = Dataset.from_list(results)
    ds_out.save_to_disk(str(out_dir / split))
    print(f"{source} ({split}): {len(ds_out)} samples saved to {out_dir / split}")
    return ds_out


for split in ["train", "validation"]:
    print(f"\nSplit: {split}")

    vp_ds = process_and_save("voxpopuli", split, VP_DIR)
    mls_ds = process_and_save("mls", split, MLS_DIR)

    combined = concatenate_datasets([vp_ds, mls_ds]).shuffle(seed=SHUFFLE_SEED)
    combined.save_to_disk(str(COMBINED_DIR / split))
    print(f"Combined: {len(combined)} samples saved to {COMBINED_DIR / split}")


ds_train = Dataset.load_from_disk(str(COMBINED_DIR / "train"))
sample_df = pd.DataFrame([{
    "tokens": len(s["labels"]),
    "transcript": processor.tokenizer.decode(s["labels"], skip_special_tokens=True),
} for s in ds_train.select(range(min(100, len(ds_train))))])

sample_path = Path("results/benchmark/02_dataset_sample.csv")
sample_path.parent.mkdir(parents=True, exist_ok=True)
sample_df.to_csv(sample_path, index=False)
print(f"\nSample preview saved to {sample_path}")
print("Done.")
