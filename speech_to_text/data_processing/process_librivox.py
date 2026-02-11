# data_processing/process_librivox.py
import os
import pandas as pd
import torch
import torchaudio
import torch.nn.functional as F
from torch.utils.data import Dataset
from utils import encode_transcript
import torchaudio.transforms as T


class LibrivoxDataset(Dataset):
    # Modificăm __init__ pentru a prelua un DataFrame direct,
    # care va include coloana 'audio_base_dir'
    def __init__(self, annotations_df, char2idx):
        self.annotations = annotations_df
        self.char2idx = char2idx

        self.mfcc_transform = T.MFCC(
            sample_rate=16000,
            n_mfcc=13,
            melkwargs={
                "n_fft": 400,
                "hop_length": 160,
                "n_mels": 23,
                "center": False
            }
        )
        self.target_sample_rate = 16000

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        row = self.annotations.iloc[idx]
        audio_filename = row[self.annotations.columns[0]]  # Prima coloană e calea relativă
        transcript = row[self.annotations.columns[1]].lower()  # A doua coloană e transcrierea
        audio_base_dir = row['audio_base_dir']  # Aici preluăm directorul de bază

        audio_path = os.path.join(audio_base_dir, audio_filename)

        waveform, sample_rate = torchaudio.load(audio_path)

        if sample_rate != self.target_sample_rate:
            resampler = T.Resample(orig_freq=sample_rate, new_freq=self.target_sample_rate)
            waveform = resampler(waveform)

        mfcc = self.mfcc_transform(waveform)
        #mfcc = mfcc.squeeze(0)  # [13, T]

        target = encode_transcript(transcript, self.char2idx)
        return mfcc, torch.tensor(target, dtype=torch.long)


def collate_librivox(batch):
    inputs, targets = zip(*batch)

    input_lengths = [mfcc.shape[1] for mfcc in inputs]
    target_lengths = [len(t) for t in targets]
    max_len = max(input_lengths)

    padded_inputs = []
    for mfcc in inputs:
        pad_length = max_len - mfcc.shape[1]
        padded_mfcc = F.pad(mfcc, (0, pad_length), value=0)
        padded_inputs.append(padded_mfcc)

    inputs_tensor = torch.stack(padded_inputs)
    targets_concat = torch.cat(targets)

    return inputs_tensor, targets_concat, torch.tensor(input_lengths, dtype=torch.long), torch.tensor(target_lengths,
                                                                                                      dtype=torch.long)