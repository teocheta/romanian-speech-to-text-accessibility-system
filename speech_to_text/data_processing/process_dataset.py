import os               #pentru lucrul cu fisiere si directoare
import pandas as pd     #pentru citire fisiere .tsv
import torch
import torchaudio       #procesare audio
import torch.nn.functional as F
import torchaudio.transforms as T  #transformari audio, ex. MFCC
from torch.utils.data import Dataset #pentru creare dataset
from torch.utils.data import DataLoader #impartire dataset in batchuri


class CommonVoiceDataset(Dataset):
    def __init__(self, audio_dir, metadata_file, transform=None, sample_rate=16000):
        """
        :param audio_dir: directorul care conține fișierele audio
        :param metadata_file: fișierul TSV cu metadatele (inclusiv transcrierea)
        :param transform: transformarea care se aplică la fișierele audio
        """
        self.audio_dir = audio_dir
        if not os.path.exists(metadata_file):
            raise FileNotFoundError(f"Fișierul {metadata_file} nu a fost găsit.")
        self.metadata = pd.read_csv(metadata_file, sep='\t')  # Se citește fișierul TSV si se incarca intr-un tabel pandas
        self.transform = transform
        self.sample_rate = sample_rate

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        # Obține informațiile pentru un fișier audio
        row = self.metadata.iloc[idx]  #se obtine randul din tabel
        filename = row['path']         #numele fisierului audio
        transcript = row['sentence']  # Folosește coloana 'sentence' pentru transcrierea fisierului

        # Calea completă a fișierului audio
        file_path = os.path.join(self.audio_dir, filename)

        # Încarcă fișierul audio
        try:
            waveform, original_sample_rate = torchaudio.load(file_path)
            # Convertim la mono si resample daca este necesar
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            if original_sample_rate != self.sample_rate:
                resampler = T.Resample(orig_freq=original_sample_rate, new_freq=self.sample_rate)
                waveform = resampler(waveform)

            # Aplică transformarea (de exemplu, MFCC)
            if self.transform:
                waveform = self.transform(waveform)

            return waveform, transcript   #waveform-> tensor pytorch cu semnalul audio

        except Exception as e:
            print(f"Eroare la încărcarea fișierului {filename}: {e}")
            return None


def collate_fn(batch):
    # Filtrăm exemplele invalide
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        raise ValueError("Batch-ul nu conține waveforms valide")

    waveforms, transcripts = zip(*batch)
    max_length = max(waveform.size(-1) for waveform in waveforms)

    padded_waveforms = []
    for waveform in waveforms:
        pad_length = max_length - waveform.size(-1)
        padded_waveform = F.pad(waveform, (0, pad_length), value=0)
        padded_waveforms.append(padded_waveform)

    waveforms_tensor = torch.stack(padded_waveforms)  # [batch, 1, time] sau [batch, n_mfcc, time]
    return waveforms_tensor, transcripts


if __name__ == "__main__":
    transform = T.MFCC(
        sample_rate=16000,
        n_mfcc=13,
        melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 23, "center": False},
    )

    audio_dir = "../cv-corpus-20.0-delta-2024-12-06/ro/clips"
    metadata_file = "../cv-corpus-20.0-delta-2024-12-06/ro/other.tsv"

    dataset = CommonVoiceDataset(audio_dir=audio_dir, metadata_file=metadata_file, transform=transform)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2, collate_fn=collate_fn)

    for waveforms, transcripts in dataloader:
        print("Forma batch-ului:", waveforms.shape)
        print("Exemplu transcriere:", transcripts[0])
        break
