"""
Training uses Seq2SeqTrainer with FP16. Checkpoints and the final model are
saved to models/whisper-spanish-finetuned/.
"""

import re
import os
import torch
import numpy as np
import pandas as pd
from dataclasses import dataclass
from pathlib import Path
from datasets import Dataset, concatenate_datasets
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
import evaluate

MODEL_ID = "openai/whisper-small"
COMBINED_DIR = Path("data/opendata/processed/combined_es")
PERSONAL_DIR = Path("data/personal/processed")
MODEL_OUT = Path("models/whisper-spanish-finetuned")
LOGS_DIR = Path("results/logs")
RESULTS_DIR = Path("results/benchmark")

MODEL_OUT.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

LEARNING_RATE = 1e-5
TRAIN_STEPS = 4000
WARMUP_STEPS = 500
EVAL_STEPS = 500
SAVE_STEPS = 500
BATCH_SIZE = 8
GRADIENT_ACCUMULATION = 2  # effective batch 16
FP16 = torch.cuda.is_available()

print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

print(f"\nLoading processor and model from {MODEL_ID} ...")
processor = WhisperProcessor.from_pretrained(MODEL_ID, language="spanish", task="transcribe")
model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID)

model.config.forced_decoder_ids = None
model.config.suppress_tokens = []
model.config.use_cache = False

model.generation_config.language = "spanish"
model.generation_config.task = "transcribe"
model.generation_config.forced_decoder_ids = None


def load_split(split: str) -> Dataset:
    path = COMBINED_DIR / split
    ds = Dataset.load_from_disk(str(path))
    print(f"Loaded combined {split}: {len(ds)} samples")

    if split == "train" and PERSONAL_DIR.exists() and (PERSONAL_DIR / "dataset_info.json").exists():
        personal = Dataset.load_from_disk(str(PERSONAL_DIR))
        personal = personal.select_columns(["input_features", "labels"])
        ds = concatenate_datasets([ds, personal]).shuffle(seed=42)
        print(f"+ personal audio: {len(personal)} samples -> total {len(ds)}")

    return ds


train_dataset = load_split("train")
eval_dataset = load_split("validation")


@dataclass
class DataCollatorSpeechSeq2Seq:
    processor: WhisperProcessor
    pad_label_id: int = -100

    def __call__(self, features: list[dict]) -> dict:
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), self.pad_label_id
        )
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


data_collator = DataCollatorSpeechSeq2Seq(processor=processor)
wer_metric = evaluate.load("wer")


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip()


def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = [normalize(s) for s in processor.batch_decode(pred_ids, skip_special_tokens=True)]
    label_str = [normalize(s) for s in processor.batch_decode(label_ids, skip_special_tokens=True)]

    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    return {"wer": round(wer, 4)}


training_args = Seq2SeqTrainingArguments(
    output_dir=str(MODEL_OUT),
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION,
    learning_rate=LEARNING_RATE,
    warmup_steps=WARMUP_STEPS,
    max_steps=TRAIN_STEPS,
    gradient_checkpointing=True,
    fp16=False,
    bf16=True,
    eval_strategy="steps",
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=SAVE_STEPS,
    eval_steps=EVAL_STEPS,
    save_total_limit=3,
    logging_steps=100,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=False,
    dataloader_num_workers=0,
)

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    processing_class=processor.feature_extractor,
)

print(f"\nTraining config:")
print(f"Steps: {TRAIN_STEPS} (warmup {WARMUP_STEPS})")
print(f"Effective batch size: {BATCH_SIZE * GRADIENT_ACCUMULATION}")
print(f"Learning rate: {LEARNING_RATE}")
print(f"FP16: {FP16}")
print(f"\nStarting training ...")
checkpoints = sorted(MODEL_OUT.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
resume = checkpoints[-1] if checkpoints else None
if resume:
    print(f"Resuming from {resume}")
trainer.train(resume_from_checkpoint=resume)

print(f"\nSaving final model to {MODEL_OUT} ...")
trainer.save_model(str(MODEL_OUT))
processor.save_pretrained(str(MODEL_OUT))

log_history = trainer.state.log_history
log_df = pd.DataFrame(log_history)
log_path = RESULTS_DIR / "04_training_log.csv"
log_df.to_csv(log_path, index=False)
print(f"Training log saved to {log_path}")

eval_rows = [row for row in log_history if "eval_wer" in row]
if eval_rows:
    best = min(eval_rows, key=lambda r: r["eval_wer"])
    print(f"\nBest checkpoint: step {best['step']}, eval WER = {best['eval_wer'] * 100:.2f}%")

print("ok")
