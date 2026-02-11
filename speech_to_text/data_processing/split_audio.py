import os
import torchaudio
import torchaudio.transforms as T

# Parametri
input_wav = "librivox/povesti/fata_mosului.wav"  # <- adaptează cu numele fișierului tău
output_dir = "../split_audio_fata_mosului"
os.makedirs(output_dir, exist_ok=True)

segment_duration = 20  # în secunde
sample_rate = 16000  # rata de eșantionare compatibilă cu modelul


# Încarcă audio și asigură format mono, 16kHz
def load_audio(filepath):
    waveform, sr = torchaudio.load(filepath)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)  # convertim la mono
    if sr != sample_rate:
        resampler = T.Resample(sr, sample_rate)
        waveform = resampler(waveform)
    return waveform.squeeze(0)  # returnăm tensor [time]

# Tăierea în segmente
waveform = load_audio(input_wav)
total_samples = waveform.size(0)
samples_per_segment = segment_duration * sample_rate

num_segments = total_samples // samples_per_segment

for i in range(num_segments):
    start = i * samples_per_segment
    end = start + samples_per_segment
    segment = waveform[start:end]
    segment_path = os.path.join(output_dir, f"segment_{i+1:03d}.wav")
    torchaudio.save(segment_path, segment.unsqueeze(0), sample_rate)
    print(f"[INFO] Salvat: {segment_path}")

print("[FINALIZAT] Fișierul audio a fost împărțit în segmente.")
