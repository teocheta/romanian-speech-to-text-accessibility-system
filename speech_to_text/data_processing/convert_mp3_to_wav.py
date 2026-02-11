import os
import torchaudio
import torchaudio.transforms as T

# Setări
input_mp3 = "librivox/multilingual_fairy_tales001_0901_librivox_64kb_mp3/12_fairytale001_fatamosului_ispirescu_nt_64kb.mp3"  # <- adaptează cu calea corectă dacă e nevoie
output_wav = "librivox/povesti/fata_mosului.wav"
sample_rate_target = 16000

# Creăm folderul dacă nu există
os.makedirs(os.path.dirname(output_wav), exist_ok=True)

# Conversie
print("[INFO] Conversie MP3 în WAV (mono, 16kHz)...")

# Încărcare MP3
waveform, sample_rate = torchaudio.load(input_mp3)

# Convertim la mono dacă este stereo
if waveform.shape[0] > 1:
    waveform = waveform.mean(dim=0, keepdim=True)

# Resample dacă e nevoie
if sample_rate != sample_rate_target:
    resampler = T.Resample(orig_freq=sample_rate, new_freq=sample_rate_target)
    waveform = resampler(waveform)

# Salvare WAV
torchaudio.save(output_wav, waveform, sample_rate_target)

print(f"[FINALIZAT] Fișierul WAV salvat la: {output_wav}")