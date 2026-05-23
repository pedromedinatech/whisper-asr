"""
Zero-shot baseline
Measures WER and CER of Whisper-small on Spanish with no fine-tuning,
evaluated on both VoxPopuli and MLS so the two datasets are directly comparable.
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
MAX_SAMPLES = 500   # increase for a more stable WER estimate
RESULTS_DIR = Path("results/benchmark")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = [
    {
        "hf_id": "facebook/voxpopuli",
        "config": "es",
        "split": "test",
        "text_col": "normalized_text",
        "label": "voxpopuli/es/test",
    },
    {
        "hf_id": "facebook/multilingual_librispeech",
        "config": "spanish",
        "split": "test",
        "text_col": "transcript",
        "label": "multilingual_librispeech/spanish/test",
    },
]


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

processor = WhisperProcessor.from_pretrained(
    MODEL_ID, language="spanish", task="transcribe"
)

model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID)
model.to(device)
model.eval()
print("Model ok.")


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)   # remove punctuation
    text = " ".join(text.split())   # collapse whitespace
    return text


def run_benchmark(hf_id: str, config: str, split: str, text_col: str, label: str) -> tuple[dict, list, list]:
    print(f"\nStreaming {label} ...")
    dataset = load_dataset(hf_id, config, split=split, streaming=True)
    dataset = dataset.cast_column("audio", Audio(decode=False))

    references = []
    hypotheses = []

    print(f"Running inference on {MAX_SAMPLES} samples ...")
    for i, sample in enumerate(dataset):
        if i >= MAX_SAMPLES:
            break

        raw = sample["audio"]
        audio_bytes = raw["bytes"] if raw["bytes"] else open(raw["path"], "rb").read()
        audio, src_sr = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=False)

        if src_sr != 16000:
            audio_tensor = torch.from_numpy(audio).unsqueeze(0)
            audio_tensor = torchaudio.functional.resample(audio_tensor, src_sr, 16000)
            audio = audio_tensor.squeeze(0).numpy()

        inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
        input_features = inputs.input_features.to(device)

        with torch.no_grad():
            predicted_ids = model.generate(
                input_features,
                language="spanish",
                task="transcribe",
            )

        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

        references.append(normalize(sample[text_col]))
        hypotheses.append(normalize(transcription))

        if (i + 1) % 50 == 0:
            partial_wer = jiwer.wer(references, hypotheses)
            print(f"[{i + 1:>4}/{MAX_SAMPLES}] partial WER: {partial_wer * 100:.2f}%")

    wer = jiwer.wer(references, hypotheses)
    cer = jiwer.cer(references, hypotheses)

    print(f"WER: {wer * 100:.2f}%  CER: {cer * 100:.2f}%  (n={len(references)})")

    row = {
        "model": MODEL_ID,
        "variant": "zero-shot",
        "dataset": label,
        "samples": len(references),
        "wer": round(wer, 4),
        "cer": round(cer, 4),
    }
    return row, references, hypotheses


rows = []
all_references = []
all_hypotheses = []
for ds in DATASETS:
    row, refs, hyps = run_benchmark(**ds)
    rows.append(row)
    all_references.extend(refs)
    all_hypotheses.extend(hyps)

combined_wer = jiwer.wer(all_references, all_hypotheses)
combined_cer = jiwer.cer(all_references, all_hypotheses)
print(f"\nCombined WER: {combined_wer * 100:.2f}%  CER: {combined_cer * 100:.2f}%  (n={len(all_references)})")
rows.append({
    "model": MODEL_ID,
    "variant": "zero-shot",
    "dataset": "combined/es/test",
    "samples": len(all_references),
    "wer": round(combined_wer, 4),
    "cer": round(combined_cer, 4),
})

results = pd.DataFrame(rows)
output_path = RESULTS_DIR / "01_baseline.csv"
results.to_csv(output_path, index=False)

print(results.to_string(index=False))
print(f"\nSaved to {output_path}")
