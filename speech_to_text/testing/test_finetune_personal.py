import os
import json
import torch
import torchaudio
import pandas as pd
from difflib import SequenceMatcher

from model.model_cnn_lstm import CNNLSTMSpeechToText
from data_processing.process_combined import CombinedDataset, collate_fn
from utils import encode_transcript, decode_prediction

# ========== CONFIG ========== #
CSV_PATH = "../data/transcriptions/combined_with_personal.csv"
MODEL_PATH = "../model/model_finetuned_personal_v2.pth"
VOCAB_PATH = "../vocab_combined.json"
RESULTS_PATH = "../results/test_finetuned_personal_results_v2.csv"

INPUT_DIM = 13
HIDDEN_DIM = 256
NUM_LAYERS = 3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========== ÎNCĂRCARE VOCABULAR ========== #
with open(VOCAB_PATH, "r", encoding="utf-8") as f:
    char2idx = json.load(f)
idx2char = {v: k for k, v in char2idx.items()}
BLANK_IDX = char2idx["_"]

# ========== ÎNCĂRCARE MODEL ========== #
model = CNNLSTMSpeechToText(
    input_dim=INPUT_DIM,
    hidden_dim=HIDDEN_DIM,
    output_dim=len(char2idx),
    num_layers=NUM_LAYERS
).to(DEVICE)

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
print("[INFO] Modelul fine-tuned pe voce personală a fost încărcat.")

# ========== TRANSFORMARE AUDIO ========== #
mfcc_transform = torchaudio.transforms.MFCC(
    sample_rate=16000,
    n_mfcc=INPUT_DIM,
    melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 23, "center": False}
)

# ========== ÎNCĂRCARE DATE ========== #
dataset = CombinedDataset(csv_path=CSV_PATH, transform=mfcc_transform)

# ========== FUNCȚIE WER ========== #
def word_error_rate(reference, hypothesis):
    ref_words = reference.split()
    hyp_words = hypothesis.split()

    m = SequenceMatcher(None, ref_words, hyp_words)
    hits = sum(block[2] for block in m.get_matching_blocks())

    if len(ref_words) == 0:
        return 1.0 if len(hyp_words) > 0 else 0.0

    error = len(ref_words) + len(hyp_words) - 2 * hits
    wer_score = error / len(ref_words)

    return wer_score

# ========== TESTARE ========== #
results = []
total_words = 0
total_errors = 0

for i in range(len(dataset)):
    item = dataset[i]
    if item is None:
        continue

    mfcc, true_transcript, filename = item
    mfcc = mfcc.unsqueeze(0).permute(0, 2, 1).to(DEVICE)

    with torch.no_grad():
        output = model(mfcc)
        output = output.squeeze(0).cpu()
        pred_indices = torch.argmax(output, dim=1).tolist()
        pred_text = decode_prediction(pred_indices, idx2char, BLANK_IDX)

    wer = word_error_rate(true_transcript, pred_text)
    total_words += len(true_transcript.split())
    total_errors += wer * len(true_transcript.split())

    print(f"[{i+1}/{len(dataset)}] {filename}")
    print(f"✓ Corect: {true_transcript}")
    print(f"✗ Pred: {pred_text}")
    print(f"WER: {wer:.4f}\n")

    results.append({
        "fisier": filename,
        "transcriere_corecta": true_transcript,
        "predictie_model": pred_text,
        "WER": round(wer, 4)
    })

# ========== REZULTAT FINAL ========== #
average_wer = total_errors / total_words if total_words else 0.0
print(f"[INFO] WER mediu pe test personal: {average_wer:.4f}")

os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
pd.DataFrame(results).to_csv(RESULTS_PATH, index=False, encoding="utf-8")
print(f"[INFO] Rezultatele au fost salvate în {RESULTS_PATH}")
