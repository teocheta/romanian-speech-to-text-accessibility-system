import torch
import pandas as pd
import os
import json
import torchaudio
from data_processing.process_dataset import CommonVoiceDataset
from model.model import SpeechToTextModel
from model.model_cnn_lstm import CNNLSTMSpeechToText
from utils import decode_prediction


# ------------ CONFIGURARE ------------
MODEL_TYPE = "cnn_lstm"  # "lstm" sau "cnn_lstm"
INPUT_DIM = 13
HIDDEN_DIM = 256
NUM_LAYERS = 3
BATCH_SIZE = 1
MODEL_PATH = "../model/model_finetune_librivox.pth"
VOCAB_PATH = "../vocab.json"
AUDIO_DIR = "../cv-corpus-20.0-delta-2024-12-06/ro/clips"
METADATA_FILE = "../cv-corpus-20.0-delta-2024-12-06/ro/other.tsv"
RESULT_PATH = "../results/test_common_voice_results_finetune.csv"

# ------------ ÎNCĂRCARE VOCABULAR ------------
with open(VOCAB_PATH, "r", encoding="utf-8") as f:
    char2idx = json.load(f)
idx2char = {int(idx): ch for ch, idx in char2idx.items()}
BLANK_IDX = char2idx['_']

# ------------ PREGĂTIRE TRANSFORMARE ------------
transform = torchaudio.transforms.MFCC(
    sample_rate=16000,
    n_mfcc=INPUT_DIM,
    melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 23, "center": False},
)

# ------------ ÎNCĂRCARE MODEL ------------
if MODEL_TYPE == "lstm":
    model = SpeechToTextModel(INPUT_DIM, HIDDEN_DIM, len(char2idx), NUM_LAYERS)
elif MODEL_TYPE == "cnn_lstm":
    model = CNNLSTMSpeechToText(INPUT_DIM, HIDDEN_DIM, len(char2idx), NUM_LAYERS)
else:
    raise ValueError("MODEL_TYPE trebuie să fie 'lstm' sau 'cnn_lstm'")
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
model.eval()

# ------------ ÎNCĂRCARE DATE ------------
dataset = CommonVoiceDataset(audio_dir=AUDIO_DIR, metadata_file=METADATA_FILE, transform=transform)

# ------------ TESTARE ȘI SALVARE ------------
results = []

for i in range(len(dataset)):
    waveform, transcript = dataset[i]
    input_tensor = waveform.unsqueeze(0).squeeze(1)

    with torch.no_grad():
        output = model(input_tensor)
        output = output.permute(1, 0, 2)
        pred_indices = torch.argmax(output, dim=2).squeeze(1).cpu().numpy()
        decoded_text = decode_prediction(pred_indices, idx2char, BLANK_IDX)
        if not decoded_text:
            decoded_text = "[EMPTY OUTPUT]"

    results.append({
        "fisier": dataset.metadata.iloc[i]['path'],
        "transcriere_corecta": transcript,
        "predictie_model": decoded_text
    })

df = pd.DataFrame(results)
os.makedirs(os.path.dirname(RESULT_PATH), exist_ok=True)
df.to_csv(RESULT_PATH, index=False, encoding="utf-8")
print(f"[FINALIZAT] Rezultatele au fost salvate în {RESULT_PATH}")
