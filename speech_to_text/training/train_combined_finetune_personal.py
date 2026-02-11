import os
import json
import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import DataLoader

from data_processing.process_combined import CombinedDataset, collate_fn
from model.model_cnn_lstm import CNNLSTMSpeechToText
from utils import encode_transcript


# ========== CONFIGURARE FINE-TUNING PE VOCEA PROPRIE ========== #
CSV_PATH = "../data/transcriptions/combined_with_personal.csv"
MODEL_SAVE_PATH = "../model/model_finetuned_personal_v2.pth"
VOCAB_LOAD_PATH = "../vocab_combined.json"

BATCH_SIZE = 4
INPUT_DIM = 13
HIDDEN_DIM = 256
NUM_LAYERS = 3
EPOCHS = 3
LEARNING_RATE = 5e-5
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

# ========== VOCABULAR (încărcare din modelul original) ========== #
with open(VOCAB_LOAD_PATH, "r", encoding="utf-8") as f:
    char2idx = json.load(f)
idx2char = {v: k for k, v in char2idx.items()}
BLANK_IDX = char2idx['_']

# ========== MODEL ========== #
model = CNNLSTMSpeechToText(
    input_dim=INPUT_DIM,
    hidden_dim=HIDDEN_DIM,
    output_dim=len(char2idx),
    num_layers=NUM_LAYERS
).to(DEVICE)

# Încărcăm modelul pre-antrenat
PRETRAINED_MODEL_PATH = "../model/model_finetuned.pth"
if os.path.exists(PRETRAINED_MODEL_PATH):
    model.load_state_dict(torch.load(PRETRAINED_MODEL_PATH, map_location=DEVICE))
    print("[INFO] Modelul pre-antrenat a fost încărcat pentru fine-tuning.")
else:
    print("[WARNING] Nu a fost găsit un model pre-antrenat. Se antrenează de la zero.")

# ========== OPTIMIZATOR ȘI FUNCȚIE DE PIERDERE ========== #
criterion = nn.CTCLoss(blank=BLANK_IDX, zero_infinity=True)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ========== LOOP DE FINE-TUNING ========== #
model.train()
for epoch in range(EPOCHS):
    epoch_loss = 0

    for batch_idx, (inputs, transcripts, _) in enumerate(dataloader):
        targets = [torch.tensor(encode_transcript(t, char2idx), dtype=torch.long) for t in transcripts]
        input_lengths = torch.full((len(inputs),), inputs.size(1) // 4, dtype=torch.long)
        target_lengths = torch.tensor([len(t) for t in targets], dtype=torch.long)
        targets_concat = torch.cat(targets)

        inputs = inputs.permute(0, 2, 1).to(DEVICE)
        targets_concat = targets_concat.to(DEVICE)
        input_lengths = input_lengths.to(DEVICE)
        target_lengths = target_lengths.to(DEVICE)

        outputs = model(inputs)
        outputs = outputs.permute(1, 0, 2)  # [T, B, C]

        loss = criterion(outputs, targets_concat, input_lengths, target_lengths)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        epoch_loss += loss.item()

        if batch_idx % 5 == 0:
            print(f"Epoca {epoch+1}, batch {batch_idx}, loss: {loss.item():.4f}")

    print(f"[INFO] Epoca {epoch+1} finalizată. Pierdere medie: {epoch_loss / len(dataloader):.4f}")

# Salvare model fine-tuned
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print("[INFO] Modelul fine-tuned pe voce personală a fost salvat.")
