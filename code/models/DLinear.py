import torch
import torch.nn as nn

class Decomposition(nn.Module):
    def __init__(self, window_size, channel_size):
        super().__init__()
        self.window_size = window_size
        # self.padding = (self.window_size - 1) / 2

        self.layer = nn.Conv1d(in_channels = channel_size, 
                               out_channels = channel_size, 
                               kernel_size = window_size, 
                               bias = False, 
                               padding = 'same',
                               dtype = torch.float64)
        
        # 이동 평균이므로 가중치가 모두 동일해야 함
        with torch.no_grad():
            self.weight = torch.ones(channel_size, 1, self.window_size) / self.window_size
            self.layer.weight.copy_(self.weight)
        self.layer.weight.requires_grad = False

    def forward(self, x):       # x.size : (batch_size, channel_size, seq_len)
        trend = self.layer(x)
        remainder = x - trend
        return trend, remainder

class DLinearModel(nn.Module):
    def __init__(self, window_size, seq_len, pred_len, channel_size):
        super().__init__()
        self.seq_len = seq_len      # history L timesteps
        self.pred_len = pred_len    # future T timesteps

        self.decomposition = Decomposition(window_size, channel_size)

        self.trend_layer = nn.Linear(self.seq_len, self.pred_len, dtype=torch.float64)
        self.remainder_layer = nn.Linear(self.seq_len, self.pred_len, dtype=torch.float64)


    def forward(self, x):
        x = x.transpose(1,2)
        trend, remainder = self.decomposition(x)            # X_t, X_s
        trend_pred = self.trend_layer(trend)                # H_t
        remainder_pred = self.remainder_layer(remainder)    # H_s
        x_hat = trend_pred + remainder_pred                 # X_hat = H_t + H_s
        return x_hat