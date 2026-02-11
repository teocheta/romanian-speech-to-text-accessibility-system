import os
import json
import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import DataLoader

from data_processing.process_combined import CombinedDataset, collate_fn
from model.model_cnn_lstm import CNNLSTMSpeechToText
from utils import build_vocab, encode_transcript

# ========== CONFIGURARE ========== #
CSV_PATH = "../data/transcriptions/combined_all_sources.csv"
MODEL_SAVE_PATH = "../model/model_combined.pth"
VOCAB_SAVE_PATH = "../vocab_combined.json"

BATCH_SIZE = 4
INPUT_DIM = 13
HIDDEN_DIM = 256
NUM_LAYERS = 3
EPOCHS = 10
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========== TRANSFORMARE AUDIO (MFCC) ========== #
mfcc_transform = torchaudio.transforms.MFCC(
    sample_rate=16000,
    n_mfcc=INPUT_DIM,
    melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 23, "center": False},
)


# ========== ÎNCĂRCARE DATE ========== #
dataset = CombinedDataset(csv_path=CSV_PATH, transform=mfcc_transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

# ========== VOCABULAR ========== #
all_transcripts = [dataset[i][1] for i in range(len(dataset)) if dataset[i] is not None]
char2idx, idx2char = build_vocab(all_transcripts)
BLANK_IDX = char2idx['_']

# Salvăm vocabularul
with open(VOCAB_SAVE_PATH, "w", encoding="utf-8") as f:
    json.dump(char2idx, f, ensure_ascii=False)

# ========== MODEL ========== #
model = CNNLSTMSpeechToText(
    input_dim=INPUT_DIM,
    hidden_dim=HIDDEN_DIM,
    output_dim=len(char2idx),
    num_layers=NUM_LAYERS
).to(DEVICE)

# Încărcăm modelul existent dacă există
if os.path.exists(MODEL_SAVE_PATH):
    try:
        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
        print("[INFO] Modelul existent a fost încărcat. Continuăm antrenarea...")
    except RuntimeError:
        print("[WARNING] Modelul existent nu este compatibil. Începem de la zero.")
else:
    print("[INFO] Se antrenează un model nou.")

# ========== OPTIMIZATOR ȘI FUNCȚIE DE PIERDERE ========== #
criterion = nn.CTCLoss(blank=BLANK_IDX, zero_infinity=True)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ========== LOOP DE ANTRENARE ========== #
model.train()
for epoch in range(EPOCHS):
    epoch_loss = 0

    for batch_idx, (inputs, transcripts, _) in enumerate(dataloader):
        # Encode transcrierile în indici
        targets = [torch.tensor(encode_transcript(t, char2idx), dtype=torch.long) for t in transcripts]
        input_lengths = torch.full((len(inputs),), inputs.size(1) // 4, dtype=torch.long)  # după 2x MaxPool
        target_lengths = torch.tensor([len(t) for t in targets], dtype=torch.long)
        targets_concat = torch.cat(targets)

        # Mutare pe dispozitiv
        # reshape inputs from [B, T, 13] to [B, 1, 13, T]
        inputs = inputs.permute(0, 2, 1)
        inputs = inputs.to(DEVICE)
        targets_concat = targets_concat.to(DEVICE)
        input_lengths = input_lengths.to(DEVICE)
        target_lengths = target_lengths.to(DEVICE)

        # Forward + loss
        outputs = model(inputs)
        outputs = outputs.permute(1, 0, 2)  # [T, B, C]

        loss = criterion(outputs, targets_concat, input_lengths, target_lengths)

        # Optimizare
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        epoch_loss += loss.item()

        if batch_idx % 10 == 0:
            print(f"Epoca {epoch+1}, batch {batch_idx}, loss: {loss.item():.4f}")

    print(f"[INFO] Epoca {epoch+1} finalizată. Pierdere medie: {epoch_loss / len(dataloader):.4f}")

    # Salvare model periodic
    if (epoch + 1) % 10 == 0:
        torch.save(model.state_dict(), f"../model/model_combined_epoch{epoch+1}.pth")
        print(f"[INFO] Model salvat: model_combined_epoch{epoch+1}.pth")

# Salvare finală
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print("[INFO] Modelul final a fost salvat.")
