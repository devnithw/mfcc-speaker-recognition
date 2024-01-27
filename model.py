import torch
from torch import nn

# Base model
class LSTMSpeakerEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, embedding_size):

        super(LSTMSpeakerEncoder, self).__init__()
        self.lstm = nn.LSTM(
            input_size,     # Number of MFCC coefficients
            hidden_size,     # Number of hidden units in each LSTM layer
            num_layers,     # Number of stacked LSTM layers
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, embedding_size) # Change the output of the LSTM to any preffered embedding size

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        embedding = self.fc(h_n[-1]) # Last layer as embedding
        return embedding