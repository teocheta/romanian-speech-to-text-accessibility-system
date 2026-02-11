import torch
import torch.nn as nn


class CNNLSTMSpeechToText(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3):
        super(CNNLSTMSpeechToText, self).__init__()

        # === Bloc CNN pentru extragerea de trăsături spațiale din spectrograma audio ===
        # Presupunem că inputul are forma [batch, 1, n_mfcc, time] — imagine 2D

        self.cnn = nn.Sequential(
            # Primul strat de convoluție: extrage trăsături locale (ex: contururi fonetice)
            nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(32),  # normalizează activările pentru stabilitate

            # Reducem dimensiunile cu pooling
            nn.MaxPool2d(kernel_size=(2, 2)),

            # Al doilea strat CNN: creștem numărul de canale pentru trăsături mai abstracte
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=(2, 2))  # din nou, reducere de dimensiuni
        )

        # === Calculăm dimensiunea de intrare pentru LSTM ===
        # După două MaxPool2D(2x2), dimensiunea pe axa MFCC scade de 4 ori
        # Canalele cresc la 64, deci inputul pentru LSTM este: (input_dim / 4) * 64
        self.lstm_input_dim = (input_dim // 4) * 64

        # === Bloc LSTM bidirecțional pentru învățarea contextului temporal ===
        self.lstm = nn.LSTM(
            input_size=self.lstm_input_dim,  # dimensiunea fiecărui vector pe timp
            hidden_size=hidden_dim,          # dimensiunea memoriei interne
            num_layers=num_layers,           # câte straturi LSTM
            batch_first=True,                # inputul este [batch, time, feature]
            bidirectional=True,              # învățăm și înainte și înapoi
            dropout=0.3                      # regularizare între straturi
        )

        # === Strat de proiecție final: transformă fiecare vector temporal în scoruri pentru caractere ===
        self.fc = nn.Linear(hidden_dim * 2, output_dim)  # *2 deoarece e bidirecțional

    def forward(self, x):
        # x are forma [batch, n_mfcc, time]
        x = x.unsqueeze(1)  # adăugăm canalul 1 → [batch, 1, n_mfcc, time]

        # === Aplicăm CNN: obținem trăsături abstracte și reducem dimensiunile ===
        x = self.cnn(x)  # ieșire: [batch, 64, n_mfcc//4, time//4]

        # === Pregătim datele pentru LSTM ===
        batch_size, channels, n_mfcc, time = x.size()

        # Rearanjăm dimensiunile: mutăm timpul ca dimensiune principală
        x = x.permute(0, 3, 1, 2)  # [batch, time, channels, n_mfcc]

        # Flatten: transformăm [channels, n_mfcc] în vector
        x = x.contiguous().view(batch_size, time, -1)  # [batch, time, channels * n_mfcc]

        # === Aplicăm LSTM ===
        x, _ = self.lstm(x)  # output: [batch, time, hidden_dim * 2]

        # === Proiecție în vocabular și normalizare log-softmax (pentru CTCLoss) ===
        x = self.fc(x)  # [batch, time, output_dim]
        return torch.log_softmax(x, dim=2)
