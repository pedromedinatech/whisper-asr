# Whisper Fine-Tuning for Informal Spanish ASR

Fine-tuning [openai/whisper-small](https://huggingface.co/openai/whisper-small) on Spanish speech data to improve transcription of informal, conversational audio. Benchmarked against zero-shot Whisper variants, Google Chirp, and ElevenLabs Scribe.

---

## Results

### Benchmark datasets (WER %)

Evaluated on VoxPopuli Spanish (spontaneous parliamentary speech) and Multilingual LibriSpeech Spanish (clean read speech). Local models: 200 samples. APIs: 100 samples.

| Model | VoxPopuli | MLS Spanish |
|---|---|---|
| Whisper-tiny (zero-shot) | 48.1 | 31.4 |
| Whisper-base (zero-shot) | 19.5 | 16.3 |
| Whisper-small (zero-shot) | 18.8 | 8.5 |
| **Whisper-small (fine-tuned)** | **9.7** | 11.8 |
| Google Chirp | 10.1 | **2.8** |
| ElevenLabs Scribe | **8.7** | 5.8 |

### Personal recordings — speaker adaptation (WER %)

Three native Spanish speakers recorded ~30 minutes of informal audio. Commercial APIs are zero-shot references on unseen speakers. The fine-tuned model was trained with speaker data.

| Model | Speaker 1 | Speaker 2 | Speaker 3 |
|---|---|---|---|
| Whisper-small (zero-shot) | 6.9 | 2.9 | 3.2 |
| Google Chirp | 3.4 | 2.2 | 2.3 |
| ElevenLabs Scribe | 4.7 | 1.8 | 7.5 |
| **Whisper-small (fine-tuned)** | **0.2** | **0.1** | **2.0** |

---

## Setup

**1. Install PyTorch** (select your CUDA version at [pytorch.org](https://pytorch.org/get-started/locally)):

```bash
# Example for CUDA 12.8 (RTX 40/50 series)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
```

**2. Install the rest of the dependencies:**

```bash
pip install -r requirements.txt
# or with conda:
conda env create -f environment.yaml && conda activate whisper-asr
```

**3. API keys** (only needed for the benchmark step, not for training):

Create a `.env` file in the project root:

```
GOOGLE_PROJECT_ID=your-gcp-project-id
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
ELEVENLABS_API_KEY=your-key
```

---

## Track A: Reproduce the benchmark

This track uses only public datasets downloaded automatically from HuggingFace. No personal audio required.

```bash
python scripts/01_baseline.py       # zero-shot WER on VoxPopuli + MLS
python scripts/02_prep_dataset.py   # downloads and preprocesses VoxPopuli + MLS
python scripts/04_finetune.py       # fine-tunes whisper-small (requires GPU)
python scripts/05_evaluate.py       # runs full benchmark, saves results/benchmark/05_model_comparison.csv
```

`02_prep_dataset.py` downloads VoxPopuli Spanish and MLS Spanish from HuggingFace automatically on first run (around 15 GB total). Set `HF_DATASETS_CACHE` if you want to control where they are stored.

---

## Track B: Run with your own speakers

This track adds personal recordings to the training data. It shows how much speaker-adaptive fine-tuning gains over general-purpose ASR.

### 1. Initialize the directory structure and metadata

```bash
python scripts/00_init_metadata.py
```

This creates `data/personal/audio/` and `data/personal/metadata.csv` pre-filled with the reference transcripts.

### 2. Record audio and drop files into `data/personal/audio/`

Record speakers reading the reference script. Each recording should be one chunk (roughly 15 seconds). Supported formats: `.wav`, `.mp3`, `.m4a`, `.mp4`. The script expects one row per audio file. Edit the file to match your recordings:

```
filename,chunk,speaker,difficulty,transcript,notes
ana_001.wav,001,ana,1,pues mira te voy a contar...,
ana_002.wav,002,ana,1,ahora intento levantarme...,
```

- `speaker`: any identifier you choose (e.g. `ana`, `carlos`)
- `difficulty`: 1 = clear, 2 = casual pace, 3 = fast or noisy
- `transcript`: pre-filled from the reference script, edit only if the speaker deviated
- `notes`: optional

The reference script (50 monologues covering daily routines, travel, technology, and language) is embedded in `00_init_metadata.py`. You can replace it with your own text.

### 3. Process personal audio

```bash
python scripts/03_personal_data.py
```

This resamples all audio to 16 kHz mono, extracts Whisper features, and saves the processed dataset to `data/personal/processed/`.

### 4. Fine-tune and evaluate

```bash
python scripts/04_finetune.py    # trains on open datasets + personal recordings
python scripts/05_evaluate.py    # benchmarks all models including personal audio
```

---

## Training details

| Parameter | Value |
|---|---|
| Base model | openai/whisper-small (244M params) |
| Training data | VoxPopuli ES + MLS ES + personal recordings |
| Hardware | NVIDIA RTX 5060 Laptop (8 GB VRAM) |
| Optimizer | AdamW, lr=1e-5, linear decay |
| Warmup steps | 500 |
| Total steps | 4,000 |
| Best checkpoint | Step 2,000 (val WER 10.51%) |
| Batch size | 16 (batch 8 + gradient accumulation 2) |
| Precision | BF16 mixed |

---

## Project structure

```
whisper/
├── scripts/
│   ├── 00_init_metadata.py       # generate metadata.csv template
│   ├── 01_baseline.py            # zero-shot evaluation
│   ├── 02_prep_dataset.py        # download + preprocess open datasets
│   ├── 03_personal_data.py       # process personal recordings
│   ├── 04_finetune.py            # fine-tuning
│   └── 05_evaluate.py            # full benchmark
├── results/benchmark/
│   ├── 01_baseline.csv           # zero-shot baseline results
│   ├── 03_personal_sample.csv    # personal audio sample inspection
│   ├── 04_training_log.csv       # loss and WER per training step
│   └── 05_model_comparison.csv   # final benchmark across all models
├── data/
│   ├── personal/
│   │   ├── audio/                # your recordings (not tracked)
│   │   └── metadata.csv          # generated by 00_init_metadata.py
│   └── opendata/processed/       # downloaded HF datasets (not tracked)
├── models/
│   └── whisper-spanish-finetuned/  # saved checkpoint (not tracked)
├── app/                          # demo app (in progress)
├── requirements.txt
└── environment.yaml
```
