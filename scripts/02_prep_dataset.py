"""
Dataset preparation: downloads VoxPopuli Spanish (train + validation),
filters bad samples, pre-computes mel spectrograms and label token IDs,
and saves to disk so the fine-tuning script loads instantly.
"""

import io
import re
import torch
import torchaudio
import soundfile as sf
from pathlib import Path
import pandas as pd
from datasets import load_dataset, Audio, Dataset
from transformers import WhisperProcessor


MODEL_ID        = "openai/whisper-small"
DATASET_ID      = "facebook/voxpopuli"
LANGUAGE_CODE   = "es"
OUTPUT_DIR      = Path("data/opendata/processed/voxpopuli_es")

MAX_TRAIN_SAMPLES = 5000  # ~7 hours; enough for Whisper-small fine-tuning
MAX_EVAL_SAMPLES  = 1000

MIN_DURATION_S = 0.5
MAX_DURATION_S = 29.5  # Whisper cap is 30 s

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Loading processor from {MODEL_ID} ...")
processor = WhisperProcessor.from_pretrained(
    MODEL_ID, language="spanish", task="transcribe"
)


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = " ".join(text.split())
    return text


def prepare_sample(sample):
    raw = sample["audio"]
    audio_bytes = raw["bytes"] if raw["bytes"] else open(raw["path"], "rb").read()
    audio, src_sr = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=False)

    if src_sr != 16000:
        audio_tensor = torch.from_numpy(audio).unsqueeze(0)
        audio_tensor = torchaudio.functional.resample(audio_tensor, src_sr, 16000)
        audio = audio_tensor.squeeze(0).numpy()

    inputs = processor(audio, sampling_rate=16000, return_tensors="np")
    sample["input_features"] = inputs.input_features[0]
    sample["labels"] = processor.tokenizer(normalize(sample["normalized_text"])).input_ids
    sample["duration"] = len(audio) / 16000.0

    return sample


def is_valid(sample) -> bool:
    if not sample.get("normalized_text", "").strip():
        return False
    dur = sample.get("duration", 0.0)
    return MIN_DURATION_S <= dur <= MAX_DURATION_S


for split, max_samples in [("train", MAX_TRAIN_SAMPLES), ("validation", MAX_EVAL_SAMPLES)]:
    print(f"\n[{split}]")

    # streaming=True + take() stops downloading after max_samples records,
    # so only the first shard(s) are fetched instead of the full 30 GB.
    ds = load_dataset(DATASET_ID, LANGUAGE_CODE, split=split, streaming=True)
    ds = ds.cast_column("audio", Audio(decode=False))
    ds = ds.take(max_samples)

    processed = []
    for i, sample in enumerate(ds):
        result = prepare_sample(sample)
        if is_valid(result):
            processed.append({
                "input_features": result["input_features"],
                "labels": result["labels"],
            })
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{max_samples} processed, {len(processed)} valid so far")

    ds_out = Dataset.from_list(processed)
    print(f"  {len(ds_out)} valid samples saved")

    out_path = OUTPUT_DIR / split
    ds_out.save_to_disk(str(out_path))
    print(f"  Saved to {out_path}")


ds_train = Dataset.load_from_disk(str(OUTPUT_DIR / "train"))

sample_df = pd.DataFrame([{
    "tokens": len(s["labels"]),
    "transcript": processor.tokenizer.decode(s["labels"], skip_special_tokens=True),
} for s in ds_train.select(range(min(100, len(ds_train))))])

sample_path = Path("results/benchmark/02_dataset_sample.csv")
sample_df.to_csv(sample_path, index=False)
print(f"\nSample saved to {sample_path}")

print("ok.", OUTPUT_DIR)
