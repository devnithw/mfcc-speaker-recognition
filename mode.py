import torch
from torch import nn
from feature_extraction import NUM_MFCC 

# Hyper parameters
LSTM_HIDDEN_SIZE = 64
LSTM_NUM_LAYERS = 3
BI_LSTM = True
FRAME_AGGREGATION_MEAN = True

# Set torch device
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Base model
class LSTMSpeakerEncoder(nn.Module):
    def __init__(self):
        super(LSTMSpeakerEncoder, self).__init__()
        self.lstm = nn.LSTM(
            input_size=NUM_MFCC,     # Number of MFCC coefficients (40)
            hidden_size=LSTM_HIDDEN_SIZE,     # Number of hidden units in each LSTM layer (64)
            num_layers=LSTM_NUM_LAYERS,     # Number of stacked LSTM layers (3)
            batch_first=True,
            bidirectional= BI_LSTM     # Whether to use a bidirectional LSTM (True/False)
        )

    def _aggregate_frames(self, batch_output):
      """
      Aggregate the output frames of the LSTM layer into a fixed-length representation
      This is the embedding vector representing the audio file
      """
      if FRAME_AGGREGATION_MEAN:
          return torch.mean(batch_output, dim=1, keepdim=False)
      else:
          return batch_output[:, -1, :]

    def forward(self, x):
        D = 2 if BI_LSTM else 1
        h0 = torch.zeros(D * LSTM_NUM_LAYERS, x.shape[0], LSTM_HIDDEN_SIZE).to(DEVICE)
        c0 = torch.zeros(D * LSTM_NUM_LAYERS, x.shape[0], LSTM_HIDDEN_SIZE).to(DEVICE)
        y, (hn, cn) = self.lstm(x, (h0, c0))
        return self._aggregate_frames(y)