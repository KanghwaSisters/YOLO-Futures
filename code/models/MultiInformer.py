import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from models.Informer.models.model import Informer
from models.Informer.utils.timefeatures import time_features

# ---- Real Time feature generator using existing timefeatures ----
class TimeFeatureGenerator(nn.Module):
    """
    기존 timefeatures.py를 활용해 실제 날짜/시간 데이터로부터 시간 특성을 생성
    Informer의 embed='timeF'와 호환되도록 freq에 맞는 시간 특성 차원을 생성.
    freq_map: {'h':4, 't':5, 's':6, 'm':1, 'a':1, 'w':2, 'd':3, 'b':3}
    """
    def __init__(self, time_dim: int, freq: str = 'd'):
        super().__init__()
        self.time_dim = time_dim
        self.freq = freq

    def forward(self, batch_size: int, seq_len: int, device: torch.device | str, dates=None):
        """
        Args:
            batch_size: 배치 크기
            seq_len: 시퀀스 길이  
            device: 디바이스
            dates: 실제 날짜 데이터 (pandas.DatetimeIndex 또는 None)
        
        Returns:
            torch.Tensor: (batch_size, seq_len, time_dim) 형태의 시간 특성
        """
        if dates is None:
            # 더미 날짜 생성 (현재 시간부터 시작)
            start_date = pd.Timestamp.now()
            dates = pd.date_range(start=start_date, periods=seq_len, freq=self.freq)
        
        # 단일 날짜 시퀀스인 경우 모든 배치에 복사
        if not isinstance(dates, list) or len(dates) != batch_size:
            dates = [dates] * batch_size
        
        batch_features = []
        for batch_idx in range(batch_size):
            batch_dates = dates[batch_idx]
            if not isinstance(batch_dates, pd.DatetimeIndex):
                batch_dates = pd.DatetimeIndex(batch_dates)
            
            # DataFrame 형태로 만들어서 time_features 함수 사용
            df_dates = pd.DataFrame({'date': batch_dates})
            
            # 기존 time_features 함수 활용 (timeenc=1로 정규화된 특성 사용)
            time_feat = time_features(df_dates, timeenc=1, freq=self.freq)
            
            # 차원 맞추기: time_feat.shape = (seq_len, actual_features)
            actual_dim = time_feat.shape[1]
            
            if actual_dim == self.time_dim:
                # 차원이 정확히 맞는 경우
                features = time_feat
            elif actual_dim > self.time_dim:
                # 차원이 더 큰 경우 앞에서부터 필요한 만큼만 사용
                features = time_feat[:, :self.time_dim]
            else:
                # 차원이 작은 경우 0으로 패딩
                padding = np.zeros((seq_len, self.time_dim - actual_dim))
                features = np.concatenate([time_feat, padding], axis=1)
            
            batch_features.append(features)
        
        # numpy array를 tensor로 변환
        time_features_tensor = torch.tensor(np.array(batch_features), dtype=torch.float32, device=device)
        
        return time_features_tensor

    def from_dataloader_timestamps(self, timestamps, batch_size, seq_len, device):
        """
        데이터로더에서 받은 timestamp 데이터를 처리
        
        Args:
            timestamps: 데이터로더에서 받은 timestamp 배열 (numpy array 또는 tensor)
            batch_size: 배치 크기
            seq_len: 시퀀스 길이
            device: 디바이스
        
        Returns:
            torch.Tensor: 시간 특성 텐서
        """
        if isinstance(timestamps, torch.Tensor):
            timestamps = timestamps.cpu().numpy()
        
        # timestamps 형태 확인 및 처리
        if len(timestamps.shape) == 1:
            # 1D 배열인 경우
            if len(timestamps) >= batch_size * seq_len:
                timestamps = timestamps[:batch_size * seq_len].reshape(batch_size, seq_len)
            else:
                # 길이가 부족한 경우 반복
                timestamps = np.tile(timestamps, (batch_size * seq_len // len(timestamps) + 1))[:batch_size * seq_len].reshape(batch_size, seq_len)
        elif len(timestamps.shape) == 2:
            # 이미 (batch_size, seq_len) 형태인 경우
            timestamps = timestamps[:batch_size, :seq_len]
        
        # 숫자형 timestamp를 datetime으로 변환
        if timestamps.dtype.kind in ['i', 'u', 'f']:  # integer, unsigned, float
            timestamps = pd.to_datetime(timestamps.flatten(), unit='s').values.reshape(batch_size, seq_len)
        
        # 각 배치별로 DatetimeIndex 생성하여 처리
        dates_list = []
        for batch_idx in range(batch_size):
            dates_list.append(pd.DatetimeIndex(timestamps[batch_idx]))
        
        return self.forward(batch_size, seq_len, device, dates=dates_list)


# ---- Value signal extractor (그대로 사용) ----
class ValueSignalExtractor(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, predictions, current_price):
        batch_size, pred_len, _ = predictions.shape
        predictions = torch.where(torch.isnan(predictions), torch.zeros_like(predictions), predictions)
        current_price = torch.where(torch.isnan(current_price), torch.ones_like(current_price), current_price)

        # 수익률
        returns = (predictions[:, -1, 0] - predictions[:, 0, 0]) / (torch.abs(predictions[:, 0, 0]) + 1e-8)
        # 변동성
        volatility = torch.std(predictions.squeeze(-1), dim=1)

        # 선형 추세
        x = torch.arange(pred_len, dtype=torch.float32, device=predictions.device).unsqueeze(0).expand(batch_size, -1)
        y = predictions.squeeze(-1)
        x_mean = x.mean(dim=1, keepdim=True)
        y_mean = y.mean(dim=1, keepdim=True)
        trend = ((x - x_mean) * (y - y_mean)).sum(dim=1) / (((x - x_mean) ** 2).sum(dim=1) + 1e-8)

        # 평균회귀 신호
        pred_mean = y.mean(dim=1)
        current_normalized = current_price.squeeze(-1) if current_price.dim() > 1 else current_price
        mean_reversion = (pred_mean - current_normalized) / (torch.abs(current_normalized) + 1e-8)

        # 모멘텀
        momentum = torch.diff(y, dim=1).mean(dim=1) if pred_len > 1 else torch.zeros(batch_size, device=predictions.device)

        signals = torch.stack([returns, volatility, trend, mean_reversion, momentum], dim=1)
        signals = torch.where(torch.isnan(signals), torch.zeros_like(signals), signals)
        return signals


# ---- MultiInformer ----
class MultiInformer(nn.Module):
    """
    3개의 서로 다른 예측 기간을 가진 Informer를 병렬로 실행
    - IMPORTANT: embed='timeF' 일 때 x_mark의 feature 차원은 freq에 의해 결정됨!
    """
    FREQ2TIME_DIM = {'h': 4, 't': 5, 's': 6, 'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}

    def __init__(self,
                 enc_in=6, dec_in=6, c_out=1,
                 seq_len=96, label_len=48,
                 d_model=512, n_heads=8, e_layers=2, d_layers=1, d_ff=2048,
                 factor=5, dropout=0.05, attn='prob', embed='timeF',
                 freq='d', activation='gelu', device='cuda'):
        super().__init__()

        # 기본 파라미터
        self.label_len = label_len
        self.device = device
        self.enc_in = enc_in
        self.dec_in = dec_in
        self.embed = embed
        self.freq = freq

        # timeF 사용 시, x_mark의 차원은 freq 기반으로 고정
        if self.embed == 'timeF':
            assert freq in self.FREQ2TIME_DIM, f"Unsupported freq '{freq}' for timeF."
            self.time_dim = self.FREQ2TIME_DIM[freq]
        else:
            # embed != 'timeF' (ex: 'fixed') 인 경우 디폴트로 4
            self.time_dim = 4

        # 실제 시간 특성 생성기 (기존 timefeatures.py 활용)
        self.time_feature_gen = TimeFeatureGenerator(time_dim=self.time_dim, freq=freq)

        # Informer 3개 (enc_in/dec_in은 입력 값(feature) 차원과 동일)
        common_kwargs = dict(
            factor=factor, d_model=d_model, n_heads=n_heads,
            e_layers=e_layers, d_layers=d_layers, d_ff=d_ff,
            dropout=dropout, attn=attn, embed=embed, freq=freq,
            activation=activation, output_attention=False,
            distil=True, mix=True, device=device
        )

        self.short_informer = Informer(
            enc_in=self.enc_in, dec_in=self.dec_in, c_out=c_out,
            seq_len=seq_len, label_len=label_len, out_len=5, **common_kwargs
        ).float()

        self.medium_informer = Informer(
            enc_in=self.enc_in, dec_in=self.dec_in, c_out=c_out,
            seq_len=seq_len, label_len=label_len, out_len=20, **common_kwargs
        ).float()

        self.long_informer = Informer(
            enc_in=self.enc_in, dec_in=self.dec_in, c_out=c_out,
            seq_len=seq_len, label_len=label_len, out_len=60, **common_kwargs
        ).float()

        self.signal_extractor = ValueSignalExtractor()

    def forward(self, x, x_mark=None, dates=None, timestamps=None):
        """
        Args:
            x: 입력 시계열 데이터 (batch_size, seq_len, features)
            x_mark: 시간 마크 (optional, 없으면 생성)
            dates: 실제 날짜 데이터 (optional, pandas.DatetimeIndex)
            timestamps: 데이터로더에서 받은 timestamp 데이터 (optional)
        """
        batch_size, seq_len, _ = x.shape

        # x_mark 생성 또는 처리
        if x_mark is None:
            if timestamps is not None:
                # 데이터로더에서 받은 timestamp 사용
                x_mark = self.time_feature_gen.from_dataloader_timestamps(
                    timestamps, batch_size, seq_len, self.device
                )
            else:
                # dates 또는 더미 날짜 사용
                x_mark = self.time_feature_gen(batch_size, seq_len, self.device, dates)
        else:
            # x_mark가 잘못된 차원으로 들어오는 경우 → time_dim으로 맞춰서 투영
            if x_mark.shape[-1] != self.time_dim:
                if x_mark.shape[-1] > self.time_dim:
                    # 차원이 더 큰 경우 앞에서부터 필요한 만큼만 사용
                    x_mark = x_mark[:, :, :self.time_dim]
                else:
                    # 차원이 작은 경우 Linear layer로 투영
                    proj = nn.Linear(x_mark.shape[-1], self.time_dim, bias=False).to(self.device)
                    x_mark = proj(x_mark)

        current_price = x[:, -1:, 0]
        all_signals = []

        try:
            # Short
            dec_inp_short = self._prepare_decoder_input(x, pred_len=5)
            y_mark_short = self._prepare_decoder_mark(x_mark, pred_len=5)
            short_pred = self.short_informer(x, x_mark, dec_inp_short, y_mark_short)
            all_signals.append(self.signal_extractor(short_pred, current_price))

            # Medium
            dec_inp_medium = self._prepare_decoder_input(x, pred_len=20)
            y_mark_medium = self._prepare_decoder_mark(x_mark, pred_len=20)
            medium_pred = self.medium_informer(x, x_mark, dec_inp_medium, y_mark_medium)
            all_signals.append(self.signal_extractor(medium_pred, current_price))

            # Long
            dec_inp_long = self._prepare_decoder_input(x, pred_len=60)
            y_mark_long = self._prepare_decoder_mark(x_mark, pred_len=60)
            long_pred = self.long_informer(x, x_mark, dec_inp_long, y_mark_long)
            all_signals.append(self.signal_extractor(long_pred, current_price))

        except Exception as e:
            print(f"MultiInformer forward 에러: {e}")
            return torch.zeros(batch_size, 15, device=self.device)

        return torch.cat(all_signals, dim=1)

    def _prepare_decoder_input(self, x, pred_len):
        batch_size, _, features = x.shape
        dec_inp_label = x[:, -self.label_len:, :]
        dec_inp_pred = torch.zeros(batch_size, pred_len, features, device=self.device)
        return torch.cat([dec_inp_label, dec_inp_pred], dim=1)

    def _prepare_decoder_mark(self, x_mark, pred_len):
        batch_size, _, time_features = x_mark.shape
        y_mark_label = x_mark[:, -self.label_len:, :]
        last_time_feature = x_mark[:, -1:, :].expand(batch_size, pred_len, time_features)
        return torch.cat([y_mark_label, last_time_feature], dim=1)


class SharedFeatureExtractor(nn.Module):
    def __init__(self,
                 value_dim=15,
                 agent_input_dim=8,
                 agent_hidden_dim=32,
                 agent_out_dim=32,
                 fusion_hidden_dim=64,
                 dropout=0.1):
        super().__init__()
        self.agent_block = AgentModel(agent_input_dim, agent_hidden_dim, agent_out_dim, dropout)
        self.fusion_fc1 = nn.Linear(value_dim + agent_out_dim, fusion_hidden_dim)
        self.fusion_relu = nn.ReLU()
        self.fusion_dropout = nn.Dropout(dropout)
        self.fusion_fc2 = nn.Linear(fusion_hidden_dim, fusion_hidden_dim)
        
    def forward(self, value_signals, agent_state):
        agent_out = self.agent_block(agent_state)
        fused = torch.cat([value_signals, agent_out], dim=1)
        x = self.fusion_fc1(fused)
        x = self.fusion_relu(x)
        x = self.fusion_dropout(x)
        x = self.fusion_fc2(x)
        return x


class AgentModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        return x


class Actor(nn.Module):
    def __init__(self, fusion_hidden_dim, action_size):
        super().__init__()
        self.actor_fc1 = nn.Linear(fusion_hidden_dim, fusion_hidden_dim)
        self.actor_fc2 = nn.Linear(fusion_hidden_dim, action_size)

    def forward(self, x):
        actor_x = F.tanh(self.actor_fc1(x))
        logits = self.actor_fc2(actor_x)
        return logits


class Critic(nn.Module):
    def __init__(self, fusion_hidden_dim):
        super().__init__()
        self.critic_fc1 = nn.Linear(fusion_hidden_dim, fusion_hidden_dim)
        self.critic_fc2 = nn.Linear(fusion_hidden_dim, 1)

    def forward(self, x):
        critic_x = F.tanh(self.critic_fc1(x))
        value = self.critic_fc2(critic_x)
        return value


class MultiInformerPV(nn.Module):
    def __init__(self, 
                 enc_in=6, dec_in=6, c_out=1,
                 seq_len=96, label_len=48,
                 d_model=512, n_heads=8, e_layers=2, d_layers=1, d_ff=2048,
                 factor=5, dropout=0.05, attn='prob', embed='timeF', 
                 activation='gelu', device='cuda', freq='d',
                 agent_input_dim=8,
                 agent_hidden_dim=32,
                 agent_out_dim=32,
                 fusion_hidden_dim=64,
                 action_size=21):
        super().__init__()
        
        self.device = device
        self.multi_informer = MultiInformer(
            enc_in=enc_in, dec_in=dec_in, c_out=c_out,
            seq_len=seq_len, label_len=label_len,
            d_model=d_model, n_heads=n_heads, e_layers=e_layers, 
            d_layers=d_layers, d_ff=d_ff, factor=factor,
            dropout=dropout, attn=attn, embed=embed, freq=freq,
            activation=activation, device=device
        )
        
        self.shared = SharedFeatureExtractor(
            value_dim=15,
            agent_input_dim=agent_input_dim,
            agent_hidden_dim=agent_hidden_dim,
            agent_out_dim=agent_out_dim,
            fusion_hidden_dim=fusion_hidden_dim,
            dropout=dropout
        )
        
        self.actor = Actor(fusion_hidden_dim, action_size)
        self.critic = Critic(fusion_hidden_dim)
        
    def forward(self, x):
        ts_data, agent_state = x
        
        if isinstance(ts_data, tuple) and len(ts_data) == 2:
            ts_values, ts_marks = ts_data
        else:
            ts_values = ts_data
            ts_marks = None
        
        value_signals = self.multi_informer(ts_values, ts_marks)
        fused_features = self.shared(value_signals, agent_state)
        
        logits = self.actor(fused_features)
        value = self.critic(fused_features)
        
        return logits, value