import torch
import torch.nn as nn
import torch.nn.functional as F


# LSTM (Long Short-Term Memory) este un tip de rețea neuronală recurentă (RNN) concepută special pentru a învăța dependențele pe termen lung din date secvențiale, cum ar fi textul, vorbirea sau seriile de timp.
class SpeechToTextModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super(SpeechToTextModel, self).__init__()

        # Creează un strat LSTM bidirecțional
        # - input_dim: dimensiunea fiecărui vector de intrare (de ex. 13 MFCC)
        # - hidden_dim: câți neuroni are fiecare direcție a LSTM-ului
        # - num_layers: câte straturi LSTM suprapuse
        # - batch_first=True: forma inputului va fi [batch_size, time, input_dim]
        # - bidirectional=True: LSTM-ul va procesa datele și înainte, și înapoi
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, bidirectional=True
        )

        # Strat linear (fully connected) care mapează output-ul LSTM la dimensiunea vocabularului
        # Deoarece e bidirecțional, output-ul LSTM are dimensiunea hidden_dim * 2
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        # x are forma [batch_size, n_mfcc, time] din preprocesare (MFCC)
        # Transpunem dimensiunile pentru a fi compatibile cu LSTM-ul:
        # Devine [batch_size, time, n_mfcc]
        x = x.transpose(1, 2)

        # Trece secvența prin LSTM
        # Output-ul `x` va avea forma [batch_size, time, hidden_dim * 2]
        # Al doilea output (hidden state) nu este folosit aici
        x, _ = self.lstm(x)

        # Proiectăm fiecare frame în spațiul vocabularului
        # Devine [batch_size, time, output_dim]
        x = self.fc(x)

        # Aplicăm log_softmax pe dimensiunea vocabularului (output_dim)
        # Acest format este necesar pentru funcția de pierdere CTC
        return torch.log_softmax(x, dim=2)


