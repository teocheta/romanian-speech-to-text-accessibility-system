import torch
import torch.nn as nn


class CNNLSTMSpeechToText(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3):
        super(CNNLSTMSpeechToText, self).__init__()

        # Bloc CNN pentru extragerea de trăsături locale din MFCC
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=(2, 2)),

            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=(2, 2))
        )

        # Calculăm dimensiunea de intrare pentru LSTM
        self.lstm_input_dim = (input_dim // 4) * (64)  # după două maxpooluri 2x2

        # LSTM bidirecțional
        self.lstm = nn.LSTM(
            input_size=self.lstm_input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )

        # Strat fully connected final
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        # x: [batch_size, n_mfcc, time]
        x = x.unsqueeze(1)  # [batch_size, 1, n_mfcc, time]

        # Aplicăm CNN
        x = self.cnn(x)

        # Pregătim pentru LSTM
        batch_size, channels, n_mfcc, time = x.size()
        x = x.permute(0, 3, 1, 2)  # [batch_size, time, channels, n_mfcc]
        x = x.contiguous().view(batch_size, time, -1)  # [batch_size, time, channels * n_mfcc]

        # Aplicăm LSTM
        x, _ = self.lstm(x)

        # Proiectăm în vocabular
        x = self.fc(x)

        return torch.log_softmax(x, dim=2)
