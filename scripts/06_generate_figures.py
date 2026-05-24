"""
Generate paper figures for the Whisper Spanish fine-tuning paper.
Outputs PDF figures to docs/figures/.
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'docs', 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

# Shared style
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 8,
    'axes.titlesize': 8,
    'axes.labelsize': 8,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'figure.dpi': 300,
})


# ── Figure 1: Training Curve ──────────────────────────────────────────────────
# Data from models/whisper-spanish-finetuned/checkpoint-4000/trainer_state.json

train_steps = [
    100, 200, 300, 400, 500, 600, 700, 800, 900, 1000,
    1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000,
    2100, 2200, 2300, 2400, 2500, 2600, 2700, 2800, 2900, 3000,
    3100, 3200, 3300, 3400, 3500, 3600, 3700, 3800, 3900, 4000,
]
train_loss = [
    2.493, 1.026, 0.650, 0.434, 0.441, 0.419, 0.300, 0.287, 0.266, 0.278,
    0.259, 0.267, 0.207, 0.140, 0.144, 0.138, 0.147, 0.143, 0.141, 0.064,
    0.078, 0.070, 0.088, 0.080, 0.089, 0.044, 0.044, 0.047, 0.043, 0.036,
    0.035, 0.023, 0.022, 0.019, 0.019, 0.019, 0.023, 0.020, 0.016, 0.015,
]

eval_steps = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000]
eval_wer   = [11.91, 12.44, 11.19, 10.51, 10.67, 10.81, 11.29, 10.94]

fig, ax1 = plt.subplots(figsize=(3.5, 2.5))

COLOR_LOSS = '#1f77b4'
COLOR_WER  = '#d62728'

ax1.plot(train_steps, train_loss, color=COLOR_LOSS, linewidth=1.3, label='Train loss')
ax1.set_xlabel('Training step')
ax1.set_ylabel('Training loss', color=COLOR_LOSS)
ax1.tick_params(axis='y', labelcolor=COLOR_LOSS)
ax1.set_ylim(bottom=0)

ax2 = ax1.twinx()
ax2.plot(eval_steps, eval_wer, color=COLOR_WER, linewidth=1.3,
         marker='o', markersize=4, linestyle='--', label='Val WER (%)')
ax2.set_ylabel('Validation WER (%)', color=COLOR_WER)
ax2.tick_params(axis='y', labelcolor=COLOR_WER)
ax2.set_ylim(9, 14)

ax1.axvline(2000, color='#555555', linestyle=':', linewidth=1.0)
ax1.text(2080, 1.6, 'best ckpt\n(step 2000)', fontsize=6, color='#555555', va='top')

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right',
           framealpha=0.9, edgecolor='#cccccc')

fig.tight_layout()
out1 = os.path.join(FIGURES_DIR, 'training_curve.pdf')
fig.savefig(out1, bbox_inches='tight')
plt.close()
print(f'Saved {out1}')


# ── Figure 2: WER Benchmark Comparison ───────────────────────────────────────
# Data from results/benchmark/05_model_comparison.csv

model_labels = [
    'Whisper\ntiny', 'Whisper\nbase', 'Whisper\nsmall',
    'Fine-tuned\n(ours)', 'Google\nChirp', 'ElevenLabs\nScribe',
]
voxpopuli_wer = [48.1, 19.5, 18.8,  9.7, 10.1,  8.7]
mls_wer       = [31.4, 16.3,  8.5, 11.8,  2.8,  5.8]

x = np.arange(len(model_labels))
width = 0.38

fig, ax = plt.subplots(figsize=(3.5, 2.6))
ax.bar(x - width / 2, voxpopuli_wer, width, label='VoxPopuli', color='#1f77b4', alpha=0.85)
ax.bar(x + width / 2, mls_wer,       width, label='MLS',        color='#ff7f0e', alpha=0.85)

ax.set_ylabel('WER (%)')
ax.set_xticks(x)
ax.set_xticklabels(model_labels, linespacing=1.1)
ax.set_ylim(0, 56)
ax.yaxis.grid(True, linewidth=0.5, alpha=0.5, linestyle='--')
ax.set_axisbelow(True)
ax.legend(framealpha=0.9, edgecolor='#cccccc')

fig.tight_layout()
out2 = os.path.join(FIGURES_DIR, 'wer_benchmark.pdf')
fig.savefig(out2, bbox_inches='tight')
plt.close()
print(f'Saved {out2}')


# ── Figure 3: Per-Speaker WER (4 key systems) ─────────────────────────────────
speaker_labels = ['Speaker 1\n(user)', 'Speaker 2\n(father)', 'Speaker 3\n(mother)']

systems = {
    'Whisper-small': [6.9, 2.9, 3.2],
    'Fine-tuned (ours)': [0.2, 0.1, 2.0],
    'Google Chirp': [3.4, 2.2, 2.3],
    'ElevenLabs Scribe': [4.7, 1.8, 7.5],
}
colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#9467bd']

x = np.arange(len(speaker_labels))
n = len(systems)
total_width = 0.75
width = total_width / n

fig, ax = plt.subplots(figsize=(3.5, 2.6))
for i, (label, values) in enumerate(systems.items()):
    offset = (i - (n - 1) / 2) * width
    ax.bar(x + offset, values, width, label=label, color=colors[i], alpha=0.85)

ax.set_ylabel('WER (%)')
ax.set_xticks(x)
ax.set_xticklabels(speaker_labels, linespacing=1.1)
ax.set_ylim(0, 10.5)
ax.yaxis.grid(True, linewidth=0.5, alpha=0.5, linestyle='--')
ax.set_axisbelow(True)
ax.legend(framealpha=0.9, edgecolor='#cccccc', loc='upper right')

fig.tight_layout()
out3 = os.path.join(FIGURES_DIR, 'wer_personal.pdf')
fig.savefig(out3, bbox_inches='tight')
plt.close()
print(f'Saved {out3}')

print('All figures generated.')
