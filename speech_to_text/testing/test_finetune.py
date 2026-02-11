import torch
import torchaudio
import pandas as pd
import os
from model.model_cnn_lstm import CNNLSTMSpeechToText
from utils import build_vocab, decode_prediction, encode_transcript
from data_processing.process_librivox import LibrivoxDataset  # Acum va folosi MFCC-urile corecte
from difflib import SequenceMatcher
import json

# Setări
COMBINED_CSV_PATH = "../data/transcriptions/combined_librivox.csv"  # Folosim CSV-ul combinat
# AUDIO_DIR nu mai este necesar aici
MODEL_PATH = "../model/model_finetune_librivox.pth"
RESULTS_CSV = "../results/test_finetune_combined.csv"  # Nume nou pentru rezultate
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# VOCAB_COMMON_VOICE_PATH pentru a încărca vocabularul original
VOCAB_COMMON_VOICE_PATH = "../vocab.json"

if not os.path.exists(VOCAB_COMMON_VOICE_PATH):
    raise FileNotFoundError(
        f"Fișierul vocabularului Common Voice nu a fost găsit la: {VOCAB_COMMON_VOICE_PATH}. Asigură-te că ai rulat train_v2.py și l-ai salvat.")

with open(VOCAB_COMMON_VOICE_PATH, "r", encoding="utf-8") as f:
    char2idx = json.load(f)
idx2char = {int(idx): ch for ch, idx in char2idx.items()}

if '_' in char2idx:
    BLANK_IDX = char2idx['_']
else:
    raise ValueError("Caracterul blank '_' nu a fost găsit în vocabular! Asigură-te că vocab.json este corect.")

print(f"[INFO] Vocabularul Common Voice încărcat. Dimensiune: {len(char2idx)}")

# Încărcare model
model = CNNLSTMSpeechToText(input_dim=13, hidden_dim=256, output_dim=len(char2idx), num_layers=3).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# Dataset (va folosi MFCC-urile corecte și datele combinate)
df_combined = pd.read_csv(COMBINED_CSV_PATH)  # Citim DataFrame-ul combinat
dataset = LibrivoxDataset(df_combined, char2idx)  # Trimitem DataFrame-ul


# Funcție pentru calculul WER (Word Error Rate)
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


# Testare
results = []
total_wer = 0
total_words = 0

print("\n--- Începe testarea ---")
for i in range(len(dataset)):
    mfcc, target_tensor = dataset[i]
    # expected_text = df_combined.iloc[i, 1] # Preluăm transcrierea originală din DataFrame-ul combinat
    # Better to get it from the dataset itself, which will guarantee consistency
    original_audio_filename = df_combined.iloc[i, 0]
    expected_text = df_combined.iloc[i, 1]

    with torch.no_grad():
        mfcc = mfcc.unsqueeze(0).to(DEVICE)
        output = model(mfcc)

        mfcc_length = mfcc.size(2)
        output_lengths = torch.tensor([mfcc_length // 4], dtype=torch.long).to(DEVICE)

        output = output.squeeze(0).cpu()

        predicted_indices = torch.argmax(output, dim=-1).tolist()

    predicted_text = decode_prediction(predicted_indices, idx2char, BLANK_IDX)
    if not predicted_text:
        predicted_text = "[EMPTY OUTPUT]"

    wer = word_error_rate(expected_text, predicted_text)
    total_wer += wer * len(expected_text.split())
    total_words += len(expected_text.split())

    print(f"Fișier:    {original_audio_filename}")
    print(f"Original:  \"{expected_text}\"")
    print(f"Prezis:    \"{predicted_text}\"")
    print(f"WER:       {wer:.4f}\n")

    results.append({
        "audio_filename": original_audio_filename,
        "original_text": expected_text,
        "predicted_text": predicted_text,
        "wer": wer
    })

# Calculează WER mediu
if total_words > 0:
    average_wer = total_wer / total_words
else:
    average_wer = 0.0

print(f"\n--- Testare finalizată ---")
print(f"WER Mediu total: {average_wer:.4f}")

# Salvează rezultatele într-un fișier CSV
results_df = pd.DataFrame(results)
results_df.to_csv(RESULTS_CSV, index=False)
print(f"Rezultatele au fost salvate în: {RESULTS_CSV}")