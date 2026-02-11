import os
import pandas as pd
import torch
import torchaudio
import torch.nn.functional as F
from torch.utils.data import Dataset


class CombinedDataset(Dataset):
    def __init__(self, csv_path, transform=None, sample_rate=16000):
        """
        :param csv_path: cale către fișierul CSV cu coloanele ['audio_base_dir', 'path', 'transcription']
        :param transform: transformare audio aplicată (ex: MFCC)
        :param sample_rate: rata de eșantionare dorită
        """
        self.data = pd.read_csv(csv_path)
        self.transform = transform
        self.sample_rate = sample_rate

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        base_dir = row["audio_base_dir"]
        file_path = row["path"]
        transcription = row["transcription"].lower()

        full_path = os.path.join(base_dir, file_path)

        try:
            waveform, original_sr = torchaudio.load(full_path)

            # convertim la mono dacă este stereo
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            # resample dacă e necesar
            if original_sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(orig_freq=original_sr, new_freq=self.sample_rate)
                waveform = resampler(waveform)

            # augmentare opțională: adăugăm zgomot alb ușor
            waveform += 0.001 * torch.randn_like(waveform)

            # transformare audio (ex: MFCC)
            if self.transform:
                features = self.transform(waveform)  # [1, n_mfcc, T]
                features = features.squeeze(0).transpose(0, 1)  # [T, n_mfcc]
            else:
                raise ValueError("Transformarea audio nu a fost specificată.")

            return features, transcription, file_path

        except Exception as e:
            print(f"[Eroare] Fișier corupt sau lipsă: {full_path} → {e}")
            return None


def collate_fn(batch):
    # Eliminăm exemplele eșuate (None)
    batch = [item for item in batch if item is not None]

    if len(batch) == 0:
        raise ValueError("Batch-ul nu conține exemple valide.")

    features, transcriptions, paths = zip(*batch)
    input_lengths = [f.shape[0] for f in features]
    max_len = max(input_lengths)

    padded_features = []
    for f in features:
        pad_len = max_len - f.shape[0]
        f_padded = F.pad(f, (0, 0, 0, pad_len))  # pad pe axa timpului
        padded_features.append(f_padded)

    inputs_tensor = torch.stack(padded_features)  # [batch, T, n_mfcc]
    return inputs_tensor, transcriptions, paths
