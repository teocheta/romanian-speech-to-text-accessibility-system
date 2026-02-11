import os
import subprocess
import torch
import torchaudio
import json

from asr_model.model_cnn_lstm import CNNLSTMSpeechToText
from asr_model.utils import decode_prediction

# ----- CONFIG -----
INPUT_DIM = 13
HIDDEN_DIM = 256
NUM_LAYERS = 3
MODEL_PATH = "model_finetuned_personal.pth"
VOCAB_PATH = "vocab_combined.json"

with open(VOCAB_PATH, "r", encoding="utf-8") as f:
    char2idx = json.load(f)
idx2char = {int(v): k for k, v in char2idx.items()}
BLANK_IDX = char2idx["_"]

model = CNNLSTMSpeechToText(INPUT_DIM, HIDDEN_DIM, len(char2idx), NUM_LAYERS)
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
model.eval()

transform = torchaudio.transforms.MFCC(
    sample_rate=16000,
    n_mfcc=INPUT_DIM,
    melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 23, "center": False},
)


def convert_to_wav(input_path: str) -> str:
    base, _ = os.path.splitext(input_path)
    wav_path = f"{base}_converted.wav"
    command = [
        'ffmpeg',
        '-y',
        '-i', input_path,
        '-ar', '16000',
        '-ac', '1',
        '-acodec', 'pcm_s16le',
        wav_path
    ]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return wav_path


def transcribe_audio(file_path: str) -> str:
    try:
        print(f"Transcriere fișier: {file_path}")
        wav_path = convert_to_wav(file_path)
        print(f"Conversie completă → {wav_path}")

        if not os.path.exists(wav_path):
            raise FileNotFoundError(f"Fișierul WAV nu a fost generat: {wav_path}")

        waveform, sr = torchaudio.load(wav_path)

        if sr != 16000:
            print(f"Resampling {sr} → 16000")
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
            waveform = resampler(waveform)

        mfcc = transform(waveform)
        input_tensor = mfcc.squeeze(0).unsqueeze(0)

        with torch.no_grad():
            output = model(input_tensor)
            output = output.permute(1, 0, 2)
            pred_indices = torch.argmax(output, dim=2).squeeze(1).cpu().numpy()
            transcript = decode_prediction(pred_indices, idx2char, BLANK_IDX)

        return transcript.strip()

    except Exception as e:
        print(f"Eroare la transcriere: {e}")
        return "[Eroare la transcriere]"

    finally:
        if os.path.exists(wav_path):
            os.remove(wav_path)
