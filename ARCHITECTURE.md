# Project Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     DATA LAYER                                  │
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────────────┐  │
│  │   Personal   │  │ Common Voice │  │  FLEURS / VoxPopuli   │  │
│  │  recordings  │  │  (Mozilla)   │  │  (Google / Facebook)  │  │
│  │  ~15+ min    │  │  ~2,300h ES  │  │  eval only            │  │
│  └──────┬───────┘  └──────┬───────┘  └───────────┬───────────┘  │
└─────────┼─────────────────┼──────────────────────┼──────────────┘
          │                 │                      │
          └────────┬────────┘                      │
                   ▼                               │
┌──────────────────────────────────────────────────┼──────────────┐
│                  PREPROCESSING LAYER             │              │
│                                                  │              │
│  1. Resample → 16kHz mono (librosa)              │              │
│  2. Filter bad samples (<1s, >30s, empty)        │              │
│  3. Normalize text (lowercase, diacritics)       │              │
│  4. Split: 80% train / 10% val / 10% test        │              │
│  5. Feature extraction (WhisperProcessor)        │              │
│     - Audio → Mel Spectrogram [80, 3000]         │              │
│     - Text  → Token IDs                          │              │
└──────────────────────┬───────────────────────────┼──────────────┘
                       │                           │
                       ▼                           ▼
┌──────────────────────────────┐  ┌───────────────────────────────┐
│      FINE-TUNING LAYER       │  │      BENCHMARK LAYER          │
│                              │  │                               │
│  Base: openai/whisper-small  │  │  Test set (same audio files)  │
│        (244M params)         │  │                               │
│                              │  │  ┌─────────────────────────┐  │
│  Seq2SeqTrainer              │  │  │ Whisper-small zero-shot │  │
│  - lr: 1e-5                  │  │  ├─────────────────────────┤  │
│  - steps: 4000               │  │  │ Whisper-small fine-tuned│  │
│  - batch: 16 (RTX 5060 8GB)  │  │  ├─────────────────────────┤  │
│  - warmup: 500               │  │  │ Google Chirp API        │  │
│  - FP16 mixed precision      │  │  ├─────────────────────────┤  │
│  - eval every 1000 steps     │  │  │ ElevenLabs ASR API      │  │
│                              │  │  └──────────┬──────────────┘  │
│  Metrics: WER (validation)   │  │             │                 │
│  Logs: TensorBoard           │  │  WER + CER comparison table   │
└──────────────┬───────────────┘  └───────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────────────────┐
│                      DEMO APP LAYER                              │
│                                                                  │
│  Gradio Interface                                                │
│                                                                  │
│  [Microphone / File Upload]                                      │
│           │                                                      │
│           ▼                                                      │
│  WhisperProcessor → mel spectrogram                              │
│           │                                                      │
│           ▼                                                      │
│  Fine-tuned WhisperForConditionalGeneration                      │
│           │                                                      │
│           ▼                                                      │
│  Transcription text                                              │
│           │                                                      │
│           ▼ (optional LLM integration)                           │
│  Claude API → Response in Spanish                                │
│           │                                                      │
│           ▼                                                      │
│  [Text output + timestamps + confidence]                         │
└──────────────────────────────────────────────────────────────────┘
```

## Component Responsibilities

| Component | Library | Role |
|-----------|---------|------|
| Data loading | `datasets` | Download and cache HF datasets |
| Audio processing | `librosa`, `soundfile`, `torchaudio` | Resample, convert, validate |
| Feature extraction | `transformers.WhisperProcessor` | Audio→mel, text→tokens |
| Model | `transformers.WhisperForConditionalGeneration` | Encoder-decoder transformer |
| Training | `transformers.Seq2SeqTrainer` + `accelerate` | Fine-tuning loop |
| Metrics | `evaluate`, `jiwer` | WER, CER calculation |
| Inference (baseline) | `openai-whisper` | Zero-shot benchmark |
| Demo | `gradio` | Web interface |

## Data Flow

```
Raw audio file (.mp3/.wav/.flac)
    → librosa.load(sr=16000, mono=True)
    → numpy array [samples]
    → WhisperFeatureExtractor
    → log-mel spectrogram [80, 3000]
    → WhisperEncoder (12 transformer blocks)
    → context vectors [1500, 768]
    → WhisperDecoder (12 transformer blocks + cross-attention)
    → token logits [vocab_size=51865]
    → argmax / beam search
    → token IDs
    → WhisperTokenizer.decode()
    → "transcribed text"
```

## Directory Structure

```
whisper/
├── environment.yaml          # conda environment definition
├── .gitignore
├── ARCHITECTURE.md           # this file
├── data/
│   ├── personal/
│   │   ├── audio/            # .wav recordings (16kHz mono)
│   │   ├── transcripts/      # .txt files (one per audio)
│   │   └── metadata.csv      # filename, duration, difficulty, transcript
│   └── opendata/
│       ├── raw/              # datasets as downloaded
│       └── processed/        # after resampling and filtering
├── dataset/                  # final HuggingFace DatasetDict
│   ├── train/
│   ├── validation/
│   └── test/
├── scripts/
│   ├── 01_baseline.py        # zero-shot WER measurement
│   ├── 02_download_data.py   # fetch Common Voice, VoxPopuli, FLEURS
│   ├── 03_prepare_dataset.py # normalize, filter, split, extract features
│   ├── 04_finetune.py        # Seq2SeqTrainer fine-tuning
│   ├── 05_benchmark.py       # compare all 4 systems
│   └── 06_app.py             # Gradio demo
├── models/
│   └── whisper-spanish-finetuned/  # saved model after training
├── results/
│   ├── benchmark/            # WER/CER tables (CSV)
│   └── logs/                 # TensorBoard training logs
├── notebooks/                # exploratory notebooks
└── app/                      # optional standalone app files
```
