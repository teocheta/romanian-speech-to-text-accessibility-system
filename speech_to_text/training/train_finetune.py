# training/train_finetune.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pandas as pd
import os
import json

from model.model_cnn_lstm import CNNLSTMSpeechToText
from utils import build_vocab, encode_transcript
from data_processing.process_librivox import LibrivoxDataset, collate_librivox

# --- Setări ---
# Acum vom folosi fișierul CSV combinat
COMBINED_CSV_PATH = "../data/transcriptions/combined_librivox.csv"
# AUDIO_DIR nu mai este necesar aici, deoarece calea completă este în CSV
MODEL_PATH = "../model/model_finetune_librivox.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Setări ajustate pentru fine-tuning
BATCH_SIZE = 4  # Poți încerca și 8 sau 16 dacă ai memorie GPU
EPOCHS = 20  # Mărește numărul de epoci
LEARNING_RATE = 1e-4  # Poți încerca 5e-5 dacă antrenamentul e instabil

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

# --- Inițializare model ---
model = CNNLSTMSpeechToText(
    input_dim=13,
    hidden_dim=256,
    output_dim=len(char2idx),
    num_layers=3
).to(DEVICE)

# --- Încărcare model existent (dacă e cazul) ---
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    print("[INFO] Modelul a fost încărcat pentru fine-tuning.")
else:
    print("[AVERTISMENT] Modelul nu a fost găsit. Se va antrena de la zero.")

# --- Dezgheață (Unfreeze) straturile LSTM ---
print("[INFO] Înghețăm straturile CNN pentru fine-tuning, antrenăm LSTM și FC.")
for name, param in model.named_parameters():
    if "lstm" in name or "fc" in name:
        param.requires_grad = True  # Asigură-te că sunt trainabile
    else:  # CNN layers
        param.requires_grad = False  # Rămân înghețate

# Verifică care parametri sunt acum trainabili
print("Parametri trainabili:")
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"- {name}")

# --- Dataset și DataLoader ---
# Citim fișierul CSV combinat
combined_annotations_df = pd.read_csv(COMBINED_CSV_PATH)
dataset = LibrivoxDataset(combined_annotations_df, char2idx)  # Trimitem DataFrame-ul
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_librivox)

# --- Funcție pierdere și optimizator ---
ctc_loss = nn.CTCLoss(blank=BLANK_IDX, zero_infinity=True)

# Antrenăm parametrii care au requires_grad=True (acum includ și LSTM)
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)

# --- Antrenare ---
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for batch_idx, (inputs, targets, input_lengths, target_lengths) in enumerate(loader):
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        input_lengths, target_lengths = input_lengths.to(DEVICE), target_lengths.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(inputs)

        output_lengths = input_lengths // 4

        outputs = outputs.permute(1, 0, 2)
        loss = ctc_loss(outputs, targets, output_lengths, target_lengths)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if batch_idx % 10 == 0:  # Printăm mai rar, pentru a nu umple consola
            print(f"Epoca {epoch + 1}, batch {batch_idx + 1}/{len(loader)}, loss: {loss.item():.4f}")

    print(f"[EPOCA {epoch + 1}] Pierdere medie: {total_loss / len(loader):.4f}")

# --- Salvare model ---
torch.save(model.state_dict(), MODEL_PATH)
print("[FINALIZAT] Fine-tuning complet. Modelul a fost salvat.")