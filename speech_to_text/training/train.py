import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data_processing.process_dataset import CommonVoiceDataset, collate_fn
from model.model import SpeechToTextModel
from utils import build_vocab, encode_transcript
import torchaudio
import os
import json


# Setări generale
BATCH_SIZE = 4
INPUT_DIM = 13
HIDDEN_DIM = 256
NUM_LAYERS = 3
EPOCHS = 50
LEARNING_RATE = 1e-3

# Încărcare date
transform = torchaudio.transforms.MFCC(
    sample_rate=16000,
    n_mfcc=INPUT_DIM,
    melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 23, "center": False},
)

audio_dir = "../cv-corpus-20.0-delta-2024-12-06/ro/clips"
metadata_file = "../cv-corpus-20.0-delta-2024-12-06/ro/other.tsv"

dataset = CommonVoiceDataset(audio_dir=audio_dir, metadata_file=metadata_file, transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

# Construim vocabularul
all_transcripts = [dataset[i][1] for i in range(len(dataset)) if dataset[i] is not None]
char2idx, idx2char = build_vocab(all_transcripts)
BLANK_IDX = char2idx['_']

# Salvăm vocabularul pentru testare ulterioară
with open("../vocab.json", "w", encoding="utf-8") as f:
    json.dump(char2idx, f, ensure_ascii=False)

# Model, loss, optimizer
model = SpeechToTextModel(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, output_dim=len(char2idx), num_layers=NUM_LAYERS)

# Încărcăm modelul dacă există deja
if os.path.exists("../model/model.pth"):
    model.load_state_dict(torch.load("../model/model.pth"))
    print("Modelul existent a fost încărcat. Continuăm antrenarea...")

ctc_loss = nn.CTCLoss(blank=BLANK_IDX, zero_infinity=True)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Antrenare
model.train()
for epoch in range(EPOCHS):
    epoch_loss = 0
    for batch_idx, (inputs, transcripts) in enumerate(dataloader):
        inputs = inputs.squeeze(1)
        targets = [encode_transcript(t, char2idx) for t in transcripts]
        input_lengths = torch.full((len(inputs),), inputs.size(2), dtype=torch.long)
        target_lengths = torch.tensor([len(t) for t in targets], dtype=torch.long)
        targets_concat = torch.cat(targets)

        outputs = model(inputs)
        outputs = outputs.permute(1, 0, 2)

        loss = ctc_loss(outputs, targets_concat, input_lengths, target_lengths)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        if batch_idx % 10 == 0:
            print(f"Epoca {epoch+1}, batch {batch_idx}, loss: {loss.item():.4f}")

    print(f"Epoca {epoch+1} finalizată. Pierdere medie: {epoch_loss / len(dataloader):.4f}")

    # Salvare intermediară la fiecare 10 epoci
    if (epoch + 1) % 10 == 0:
        torch.save(model.state_dict(), f"model_epoch_{epoch + 1}.pth")
        print(f"Modelul a fost salvat ca model_epoch_{epoch + 1}.pth")

# Salvare model antrenat
torch.save(model.state_dict(), "../model/model.pth")
print("Modelul a fost salvat ca model.pth")
