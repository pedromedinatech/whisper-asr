"""
Comprehensive benchmark: Whisper variants vs Google Chirp vs ElevenLabs Scribe.

Models
------
  whisper-tiny            zero-shot (local)
  whisper-base            zero-shot (local)
  whisper-small           zero-shot (local)
  whisper-small-finetuned fine-tuned on VoxPopuli+MLS+personal (local)
  google-chirp            Google Cloud Speech-to-Text v2 (API)
  elevenlabs-scribe       ElevenLabs Speech-to-Text scribe_v1 (API)

Datasets
--------
  VoxPopuli es/test       BENCHMARK_SAMPLES samples
  MLS Spanish/test        BENCHMARK_SAMPLES samples
  Personal (yo)           all — speaker is the user himself
  Personal (pedro)        all — speaker is the user's father
  Personal (mama)         all — speaker is the user's mother
                          (add her audios to data/personal/audio/ with
                           speaker=mama in metadata.csv before running)

API keys
--------
  Set GOOGLE_PROJECT_ID and GOOGLE_APPLICATION_CREDENTIALS env vars for Chirp.
  Set ELEVENLABS_API_KEY env var for ElevenLabs.
  Models are skipped (with a warning) if credentials are missing.

Output
------
  results/benchmark/05_model_comparison.csv
"""

import io
import os
import re
import wave
import struct
import torch
import numpy as np
import librosa
import soundfile as sf
import torchaudio
import jiwer
import pandas as pd
from pathlib import Path
from datasets import load_dataset, Audio
from transformers import WhisperProcessor, WhisperForConditionalGeneration

BENCHMARK_SAMPLES = 200
API_SAMPLES = 100       # cap for paid APIs to limit cost
RESULTS_DIR = Path("results/benchmark")
PERSONAL_META = Path("data/personal/metadata.csv")
PERSONAL_AUDIO_DIR = Path("data/personal/audio")
FINETUNED_MODEL = Path("models/whisper-spanish-finetuned")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

BENCHMARK_DATASETS = [
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
        "label": "mls/spanish/test",
    },
]

# speaker value in metadata.csv -> human-readable label
PERSONAL_SPEAKERS = {
    "yo": "personal/yo (user)",
    "pedro": "personal/pedro (father)",
    "isabel": "personal/isabel (mother)",
}

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return " ".join(text.split())


def audio_to_wav_bytes(audio: np.ndarray, sr: int = 16000) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Whisper (local) helpers
# ---------------------------------------------------------------------------

def load_whisper(model_id_or_path: str) -> tuple:
    path = Path(model_id_or_path)
    source = str(path) if path.exists() else model_id_or_path
    print(f"  Loading {source} ...")
    processor = WhisperProcessor.from_pretrained(source, language="spanish", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(source)
    model.to(device)
    model.eval()
    return processor, model


def whisper_transcribe(audio: np.ndarray, processor, model) -> str:
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
    features = inputs.input_features.to(device)
    with torch.no_grad():
        ids = model.generate(features, language="spanish", task="transcribe")
    return processor.batch_decode(ids, skip_special_tokens=True)[0]


# ---------------------------------------------------------------------------
# Google Chirp helper
# ---------------------------------------------------------------------------

def chirp_transcribe(audio_bytes: bytes, project_id: str) -> str:
    from google.cloud.speech_v2 import SpeechClient
    from google.cloud.speech_v2.types import cloud_speech
    from google.api_core.client_options import ClientOptions

    client = SpeechClient(
        client_options=ClientOptions(api_endpoint="us-central1-speech.googleapis.com")
    )
    config = cloud_speech.RecognitionConfig(
        auto_decoding_config=cloud_speech.AutoDetectDecodingConfig(),
        language_codes=["es-ES"],
        model="chirp",
    )
    request = cloud_speech.RecognizeRequest(
        recognizer=f"projects/{project_id}/locations/us-central1/recognizers/_",
        config=config,
        content=audio_bytes,
    )
    response = client.recognize(request=request)
    return " ".join(r.alternatives[0].transcript for r in response.results if r.alternatives)


# ---------------------------------------------------------------------------
# ElevenLabs Scribe helper
# ---------------------------------------------------------------------------

def elevenlabs_transcribe(audio_bytes: bytes, api_key: str) -> str:
    import httpx

    response = httpx.post(
        "https://api.elevenlabs.io/v1/speech-to-text",
        headers={"xi-api-key": api_key},
        files={"file": ("audio.wav", audio_bytes, "audio/wav")},
        data={"model_id": "scribe_v1", "language_code": "es"},
        timeout=30.0,
    )
    response.raise_for_status()
    return response.json().get("text", "")


# ---------------------------------------------------------------------------
# Generic evaluation runners
# ---------------------------------------------------------------------------

def eval_hf_whisper(ds_cfg: dict, processor, model, max_samples: int) -> dict:
    label = ds_cfg["label"]
    print(f"    {label} (n={max_samples}) ...")
    dataset = load_dataset(ds_cfg["hf_id"], ds_cfg["config"], split=ds_cfg["split"], streaming=True)
    dataset = dataset.cast_column("audio", Audio(decode=False))

    refs, hyps = [], []
    for i, sample in enumerate(dataset):
        if i >= max_samples:
            break
        raw = sample["audio"]
        audio_bytes = raw["bytes"] if raw["bytes"] else open(raw["path"], "rb").read()
        audio, src_sr = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=False)
        if src_sr != 16000:
            t = torch.from_numpy(audio).unsqueeze(0)
            audio = torchaudio.functional.resample(t, src_sr, 16000).squeeze(0).numpy()

        refs.append(normalize(sample[ds_cfg["text_col"]]))
        hyps.append(normalize(whisper_transcribe(audio, processor, model)))

        if (i + 1) % 50 == 0:
            print(f"      [{i+1}/{max_samples}] WER so far: {jiwer.wer(refs, hyps)*100:.2f}%")

    wer = jiwer.wer(refs, hyps)
    cer = jiwer.cer(refs, hyps)
    print(f"      WER: {wer*100:.2f}%  CER: {cer*100:.2f}%")
    return {"dataset": label, "samples": len(refs), "wer": round(wer, 4), "cer": round(cer, 4)}


def eval_hf_api(ds_cfg: dict, transcribe_fn, max_samples: int) -> dict:
    label = ds_cfg["label"]
    print(f"    {label} (n={max_samples}) ...")
    dataset = load_dataset(ds_cfg["hf_id"], ds_cfg["config"], split=ds_cfg["split"], streaming=True)
    dataset = dataset.cast_column("audio", Audio(decode=False))

    refs, hyps = [], []
    for i, sample in enumerate(dataset):
        if i >= max_samples:
            break
        raw = sample["audio"]
        audio_bytes_raw = raw["bytes"] if raw["bytes"] else open(raw["path"], "rb").read()
        audio, src_sr = sf.read(io.BytesIO(audio_bytes_raw), dtype="float32", always_2d=False)
        if src_sr != 16000:
            t = torch.from_numpy(audio).unsqueeze(0)
            audio = torchaudio.functional.resample(t, src_sr, 16000).squeeze(0).numpy()

        wav_bytes = audio_to_wav_bytes(audio)
        try:
            hyp = transcribe_fn(wav_bytes)
        except Exception as exc:
            print(f"      WARNING sample {i}: {exc}")
            hyp = ""

        refs.append(normalize(sample[ds_cfg["text_col"]]))
        hyps.append(normalize(hyp))

        if (i + 1) % 25 == 0:
            print(f"      [{i+1}/{max_samples}] WER so far: {jiwer.wer(refs, hyps)*100:.2f}%")

    wer = jiwer.wer(refs, hyps)
    cer = jiwer.cer(refs, hyps)
    print(f"      WER: {wer*100:.2f}%  CER: {cer*100:.2f}%")
    return {"dataset": label, "samples": len(refs), "wer": round(wer, 4), "cer": round(cer, 4)}


def eval_personal_whisper(speaker: str, label: str, processor, model) -> dict | None:
    if not PERSONAL_META.exists():
        return None
    meta = pd.read_csv(PERSONAL_META)
    subset = meta[meta["speaker"] == speaker]
    if subset.empty:
        print(f"    No entries for speaker={speaker}, skipping.")
        return None

    refs, hyps, missing = [], [], 0
    for _, row in subset.iterrows():
        p = PERSONAL_AUDIO_DIR / str(row["filename"])
        if not p.exists():
            missing += 1
            continue
        audio, _ = librosa.load(str(p), sr=16000, mono=True, dtype=np.float32)
        refs.append(normalize(str(row["transcript"])))
        hyps.append(normalize(whisper_transcribe(audio, processor, model)))

    if not refs:
        return None
    if missing:
        print(f"    Warning: {missing} files missing.")

    wer = jiwer.wer(refs, hyps)
    cer = jiwer.cer(refs, hyps)
    print(f"      {label}: WER {wer*100:.2f}%  CER {cer*100:.2f}%  (n={len(refs)})")
    return {"dataset": label, "samples": len(refs), "wer": round(wer, 4), "cer": round(cer, 4)}


def eval_personal_api(speaker: str, label: str, transcribe_fn) -> dict | None:
    if not PERSONAL_META.exists():
        return None
    meta = pd.read_csv(PERSONAL_META)
    subset = meta[meta["speaker"] == speaker]
    if subset.empty:
        return None

    refs, hyps, missing = [], [], 0
    for _, row in subset.iterrows():
        p = PERSONAL_AUDIO_DIR / str(row["filename"])
        if not p.exists():
            missing += 1
            continue
        audio, _ = librosa.load(str(p), sr=16000, mono=True, dtype=np.float32)
        wav_bytes = audio_to_wav_bytes(audio)
        try:
            hyp = transcribe_fn(wav_bytes)
        except Exception as exc:
            print(f"      WARNING {row['filename']}: {exc}")
            hyp = ""
        refs.append(normalize(str(row["transcript"])))
        hyps.append(normalize(hyp))

    if not refs:
        return None

    wer = jiwer.wer(refs, hyps)
    cer = jiwer.cer(refs, hyps)
    print(f"      {label}: WER {wer*100:.2f}%  CER {cer*100:.2f}%  (n={len(refs)})")
    return {"dataset": label, "samples": len(refs), "wer": round(wer, 4), "cer": round(cer, 4)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# Set to True to skip local Whisper inference and only re-run API models.
# Existing Whisper rows are loaded from the CSV and preserved.
APIS_ONLY = True

rows = []

if APIS_ONLY and (RESULTS_DIR / "05_model_comparison.csv").exists():
    existing = pd.read_csv(RESULTS_DIR / "05_model_comparison.csv")
    rows = existing[~existing["model"].isin(["google-chirp", "elevenlabs-scribe"])].to_dict("records")
    print(f"APIS_ONLY mode: loaded {len(rows)} existing Whisper rows, re-running APIs only.")

# --- Local Whisper models ---------------------------------------------------

whisper_models = [
    ("openai/whisper-tiny", "zero-shot"),
    ("openai/whisper-base", "zero-shot"),
    ("openai/whisper-small", "zero-shot"),
]
if FINETUNED_MODEL.exists():
    whisper_models.append((str(FINETUNED_MODEL), "fine-tuned"))
else:
    print(f"WARNING: fine-tuned model not found at {FINETUNED_MODEL}")

for model_source, variant in ([] if APIS_ONLY else whisper_models):
    model_label = Path(model_source).name if Path(model_source).exists() else model_source
    print(f"\n{'='*60}")
    print(f"Model: {model_label} ({variant})")
    print("=" * 60)

    processor, model = load_whisper(model_source)

    for ds_cfg in BENCHMARK_DATASETS:
        result = eval_hf_whisper(ds_cfg, processor, model, BENCHMARK_SAMPLES)
        rows.append({"model": model_label, "variant": variant, **result})

    for speaker, label in PERSONAL_SPEAKERS.items():
        result = eval_personal_whisper(speaker, label, processor, model)
        if result:
            rows.append({"model": model_label, "variant": variant, **result})

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# --- Google Chirp -----------------------------------------------------------

google_project = os.environ.get("GOOGLE_PROJECT_ID", "")
if not google_project:
    print("\nSkipping Google Chirp: GOOGLE_PROJECT_ID not set.")
else:
    try:
        from google.cloud.speech_v2 import SpeechClient
        print(f"\n{'='*60}")
        print("Model: google-chirp (API)")
        print("=" * 60)

        def chirp_fn(wav_bytes: bytes) -> str:
            return chirp_transcribe(wav_bytes, google_project)

        for ds_cfg in BENCHMARK_DATASETS:
            result = eval_hf_api(ds_cfg, chirp_fn, API_SAMPLES)
            rows.append({"model": "google-chirp", "variant": "api", **result})

        for speaker, label in PERSONAL_SPEAKERS.items():
            result = eval_personal_api(speaker, label, chirp_fn)
            if result:
                rows.append({"model": "google-chirp", "variant": "api", **result})

    except ImportError:
        print("Skipping Google Chirp: google-cloud-speech not installed.")
        print("  Install with: pip install google-cloud-speech")

# --- ElevenLabs Scribe ------------------------------------------------------

el_api_key = os.environ.get("ELEVENLABS_API_KEY", "")
if not el_api_key:
    print("\nSkipping ElevenLabs: ELEVENLABS_API_KEY not set.")
else:
    try:
        import httpx
        print(f"\n{'='*60}")
        print("Model: elevenlabs-scribe (API)")
        print("=" * 60)

        def elevenlabs_fn(wav_bytes: bytes) -> str:
            return elevenlabs_transcribe(wav_bytes, el_api_key)

        for ds_cfg in BENCHMARK_DATASETS:
            result = eval_hf_api(ds_cfg, elevenlabs_fn, API_SAMPLES)
            rows.append({"model": "elevenlabs-scribe", "variant": "api", **result})

        for speaker, label in PERSONAL_SPEAKERS.items():
            result = eval_personal_api(speaker, label, elevenlabs_fn)
            if result:
                rows.append({"model": "elevenlabs-scribe", "variant": "api", **result})

    except ImportError:
        print("Skipping ElevenLabs: httpx not installed.")
        print("  Install with: pip install httpx")

# --- Save results -----------------------------------------------------------

df = pd.DataFrame(rows)
output_path = RESULTS_DIR / "05_model_comparison.csv"
df.to_csv(output_path, index=False)

print(f"\n{'='*60}")
print("FINAL RESULTS")
print("=" * 60)
print(df.to_string(index=False))
print(f"\nSaved to {output_path}")
