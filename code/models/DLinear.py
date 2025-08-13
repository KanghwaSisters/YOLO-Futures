import numpy as np
import random
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import time

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

############# DLinear 모델 부분
class Decomposition(nn.Module):
    def __init__(self, window_size, pred_feature, device, trainable=True):
        super().__init__()
        self.window_size = window_size
        self.pred_feature = pred_feature
        self.device = device
        self.padding = self.window_size // 2

        self.layer = nn.Conv1d(in_channels = self.pred_feature, 
                               out_channels = self.pred_feature, 
                               kernel_size = self.window_size, 
                               bias = False, 
                               padding = self.padding,
                               padding_mode='replicate',
                               groups = self.pred_feature,
                               dtype = torch.float32,
                               device = self.device)
        
        if not trainable:
            # 이동 평균이므로 가중치가 모두 동일해야 함
            weight = torch.ones(self.pred_feature, 1, self.window_size, device = self.device) / self.window_size
            self.register_buffer('weight', weight)

            with torch.no_grad():
                self.layer.weight.copy_(weight)
            self.layer.weight.requires_grad = False

    def forward(self, x):       # x.size : (batch_size, channel_size, seq_len)
        trend = self.layer(x)
        remainder = x - trend
        return trend, remainder


class DLinearModel(nn.Module):
    def __init__(self, window_size, seq_len, pred_len, pred_feature, device):
        super().__init__()
        self.window_size = window_size
        self.seq_len = seq_len      # history L timesteps
        self.pred_len = pred_len    # future T timesteps
        self.pred_feature = pred_feature
        self.device = device

        self.decomposition = Decomposition(self.window_size, self.pred_feature, self.device, trainable=False)

        self.trend_layer = nn.Linear(self.seq_len, self.pred_len, dtype=torch.float32, device=self.device)
        self.remainder_layer = nn.Linear(self.seq_len, self.pred_len, dtype=torch.float32, device=self.device)


    def forward(self, x):
        x_pred = x[:, :self.pred_feature]    # 예측하는 feature
        x_rest = x[:, :self.pred_feature]    # 추가 정보

        trend, remainder = self.decomposition(x_pred)       # 시계열 분해 (trend, remainder)
        trend = torch.cat([trend, x_rest], dim=1)           # X_t
        remainder = torch.cat([remainder, x_rest], dim=1)   # X_s

        trend_pred = self.trend_layer(trend)                # H_t
        remainder_pred = self.remainder_layer(remainder)    # H_s
        x_hat = trend_pred + remainder_pred                 # X_hat = H_t + H_s
        return trend_pred, remainder_pred, x_hat

################### train 관련
class TimeSeriesDataset(Dataset):
    def __init__(self, df, seq_len=80, pred_len=15, scale_factor=1.0, train=True, mean=None, sd=None):
        self.data = df.values.astype(np.float32)
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.scale_factor = scale_factor

        if train:
            self.mean = np.mean(self.data, axis=0)
            self.sd = np.std(self.data, axis=0)
        else:
            self.mean = mean
            self.sd = mean

        self.norm_data = ((self.data - self.mean) / self.sd) * self.scale_factor
        
    def __getitem__(self, idx):
        x = self.norm_data[idx : idx + self.seq_len].transpose()
        y = self.norm_data[idx + self.seq_len : idx + self.seq_len + self.pred_len].transpose()

        # x = self.data[idx : idx + self.seq_len].transpose()
        # y = self.data[idx + self.seq_len : idx + self.seq_len + self.pred_len].transpose()
        return x, y

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len

    def inverse_transform(self, data, features):
        data = np.array(data)
        return data * self.sd[:features] + self.mean[:features]
        # return data


class DLinearAgent:
    def __init__(self, data, batch_size, train_rate, test_rate, time_section,
                 window_size, seq_len, pred_len, pred_feature, device,
                 discrete_var:bool, threshold_rate, 
                 lr, alpha, beta, gamma, lmbd, scale_factor,
                 folder_name):
        self.data = data
        self.batch_size = batch_size
        self.train_rate = train_rate
        self.test_rate = test_rate
        self.time_section = time_section

        self.window_size = window_size
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.pred_feature = pred_feature
        self.device = device

        self.discrete_var = discrete_var
        self.lr = lr
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.lmbd = lmbd
        self.scale_factor = scale_factor

        self.total_len = len(data)

        self._create_dataloader()
        self._create_model()
        self.mse_loss = nn.MSELoss().to(self.device)

        self.loss_list = []

        self.threshold = self._calculate_threshold(threshold_rate)
        self.file_name = f'model_t{self.time_section}_s{self.seq_len}p{self.pred_len}w{self.window_size}'
        self.folder_name = folder_name
        self.train_mode = False

        self.loss_type_list = {1: 'hybrid_mse', 
                               2: 'restrict_trend',
                               3: 'restrict_remainder',
                               4: 'restrict_both'}
        self.loss_type = self.loss_type_list[1]
        print(f"loss type list: {self.loss_type_list}")

    def _create_dataloader(self):
        train_end_idx = int(self.total_len * self.train_rate)
        test_start_idx = int(self.total_len * (1-self.test_rate))

        self.train_dataset = TimeSeriesDataset(df=self.data.iloc[:train_end_idx], 
                                                seq_len=self.seq_len, pred_len=self.pred_len,
                                                scale_factor=self.scale_factor, train=True)
        self.valid_dataset = TimeSeriesDataset(df=self.data.iloc[train_end_idx:test_start_idx], 
                                                seq_len=self.seq_len, pred_len=self.pred_len,
                                                scale_factor=self.scale_factor, train=False, mean=self.train_dataset.mean, sd=self.train_dataset.sd)
        self.test_dataset = TimeSeriesDataset(df=self.data.iloc[test_start_idx:], 
                                                seq_len=self.seq_len, pred_len=self.pred_len,
                                                scale_factor=self.scale_factor, train=False, mean=self.train_dataset.mean, sd=self.train_dataset.sd)

        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.valid_loader = DataLoader(self.valid_dataset, batch_size=self.batch_size, shuffle=False)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)
        print("creat datasets, dataloaders")

    def _create_model(self):
        self.model_list = []

        if self.discrete_var:
            # 0: low variability / 1: high variability
            for _ in range(2):
                model = DLinearModel(window_size=self.window_size, seq_len=self.seq_len, pred_len=self.pred_len, pred_feature=self.pred_feature, device=self.device)
                optimizer = optim.Adam(model.parameters(), lr=self.lr)
                self.model_list.append((model, optimizer))
            print("creat models")

        else:
            model = DLinearModel(window_size=self.window_size, seq_len=self.seq_len, pred_len=self.pred_len, pred_feature=self.pred_feature, device=self.device)
            optimizer = optim.Adam(model.parameters(), lr=self.lr)
            self.model_list.append((model, optimizer))
            print("creat model")

    def _calculate_threshold(self, rate):
        std_list = []
        for idx in range(len(self.train_dataset)):
            x, _ = self.train_dataset[idx]
            std = x.std(axis=1)[0]
            std_list.append(std)
        std_array = np.array(std_list)
        threshold = np.percentile(std_array, rate*10)  # 상위 10%

        return threshold

    def _loss_ftn(self, pred_trend, pred_remainder, pred_y, y, prev_x):
        # 단순 mse loss
        mse_loss = self.mse_loss(pred_y, y)
        
        # 기울기 loss
        slope_pred = pred_y[:, -1, :] - prev_x
        slope_true = y[:, -1, :] - prev_x
        slope_loss = self.mse_loss(slope_pred, slope_true)

        # 차분 loss
        pred_with_prev = torch.cat([prev_x, pred_y], dim=2)
        true_with_prev = torch.cat([prev_x, y], dim=2)
        diff_pred = torch.diff(pred_with_prev, dim=2)
        diff_true = torch.diff(true_with_prev, dim=2)
        diff_loss = self.mse_loss(diff_pred, diff_true)

        hybrid_loss = mse_loss * self.alpha + slope_loss * self.beta + diff_loss * self.gamma

        if self.loss_type == 'hybrid_mse':
            return hybrid_loss

        trend_diff = pred_trend[:, :, 1:] - pred_trend[:, :, :-1]
        smoothness_loss = abs(torch.mean(trend_diff ** 2))
        if self.loss_type == 'restrict_trend':
            return hybrid_loss + smoothness_loss * self.lmbd

        zero_mean_loss = abs(torch.mean(pred_remainder))
        if self.loss_type == 'restrict_remainder':
            return hybrid_loss + zero_mean_loss * self.lmbd

        if self.loss_type == 'restrict_both':
            return hybrid_loss + (smoothness_loss + zero_mean_loss) * self.lmbd

    def print_info(self):
        print(f"[Info] device:{self.device} | loss ftn:{self.loss_type}\n\
            time section:{self.time_section} | lr:{self.lr} | discrete by var:{self.discrete_var}\n\
            batch size:{self.batch_size} | train rate:{self.train_rate} | test rate:{self.test_rate}\n\
            seq len: {self.seq_len} | pred len: {self.pred_len} | window size: {self.window_size}\n\
            -------------------------------")

    def train(self, epoch, print_freq, valid_freq, save_freq, visualizing_num,
                loss_idx=1):
        self.loss_type = self.loss_type_list[loss_idx]
        self.print_info()
        time_list = []

        print("<< start train >>")
        for e in range(epoch):
            current_time = time.time()
            epoch_loss = 0
            self.train_mode = True

            for model, _ in self.model_list:
                model.train()

            for x, y in self.train_loader:
                loss = self._train_one_step(x, y)
                epoch_loss += loss

            self.loss_list.append(epoch_loss / len(self.train_loader))
            time_list.append(time.time() - current_time)

            if (e+1) % print_freq == 0:
                print(f"epoch [{e+1}/{epoch}] | loss: {np.round(np.mean(self.loss_list[-print_freq:]), 3)} -- time: {np.round(np.mean(time_list), 3)}")
                time_list = []

            if (e+1) % save_freq == 0:
                self._save_current_model()
            
            if (e+1) % valid_freq == 0:
                # valid
                self.train_mode = False
                self.test("valid", visualizing_num)

        # test
        self.train_mode = False
        self.test("test", visualizing_num)
        self._visualizing_loss_curve()
        print("<< finish train >>")

    def _train_one_step(self, x, y):
        if self.discrete_var:
            # 변동성에 따라 
            x_stds = torch.std(x[:,0], axis=1)
            high_idxs = torch.where(x_stds > self.threshold)[0]
            if len(high_idxs) == 0:
                model, optimizer = self.model_list[0]
                loss = self._train_block(x, y, model, optimizer)

            elif len(high_idxs) == len(x):
                model, optimizer = self.model_list[1]
                loss = self._train_block(x, y, model, optimizer)

            else:
                normal_x = x[~high_idxs, :, :]
                normal_y = y[~high_idxs, :, :]
                normal_model, normal_optimizer = self.model_list[0]
                low_loss = self._train_block(normal_x, normal_y, normal_model, normal_optimizer)

                high_x = x[high_idxs, :, :]
                high_y = y[high_idxs, :, :]
                high_model, high_optimizer = self.model_list[1]
                high_loss = self._train_block(high_x, high_y, high_model, high_optimizer)
                    
                loss = low_loss + high_loss

        else:
            model, optimizer = self.model_list[0]
            loss = self._train_block(x, y, model, optimizer)

        return loss

    def _train_block(self, x, y, model, optimizer):
        x = x.to(self.device)
        y = y.to(self.device)
        pred_trend, pred_remainder, pred_y = model.forward(x)

        loss = self._loss_ftn(pred_trend, pred_remainder, pred_y[:,:self.pred_feature], y[:, :self.pred_feature], x[:,:self.pred_feature,-1:])
        
        if not self.train_mode:
            return loss.to('cpu').item()

        else:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            return loss.to('cpu').item()

    def _save_current_model(self):
        print("save model")
        for i in range(len(self.model_list)):
            model, _ = self.model_list[i]
            with open(f'{self.folder_name}/{self.file_name}_{i}.pkl', 'wb') as f:
                pickle.dump(model.state_dict(), f)

    def _visualizing(self, dataset, num):
        for _ in range(num):
            idx = random.randint(0, len(dataset) - 1)
            x, y = dataset[idx]

            x_torch = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(self.device)
            if self.discrete_var:
                is_high = np.std(x) > self.threshold
                model, _ = self.model_list[int(is_high)]
            else:
                model, _ = self.model_list[0]
            pred_trend, _, pred_y = model.forward(x_torch)
            plot_data = np.concatenate([x[:self.pred_feature,:], y[:self.pred_feature,:]], axis=1)
            plot_data = self.train_dataset.inverse_transform(plot_data, self.pred_feature).reshape(-1)
            torch_plot_data = torch.tensor(plot_data).unsqueeze(0).to(self.device)
            trend, _ = model.decomposition.forward(torch_plot_data)
            pred_trend = pred_trend.squeeze(0)
            pred_trend = pred_trend[:self.pred_feature,:].to('cpu').detach().numpy()
            pred_y = pred_y.squeeze(0)
            pred_y = pred_y[:self.pred_feature,:].to('cpu').detach().numpy()
            trend_data = self.train_dataset.inverse_transform(pred_trend, self.pred_feature).reshape(-1)
            pred_data = self.train_dataset.inverse_transform(pred_y, self.pred_feature).reshape(-1)

            plt.plot(range(self.seq_len + self.pred_len), plot_data, color='blue')
            plt.plot(range(self.seq_len + self.pred_len), trend.cpu().squeeze().squeeze().tolist(), color='green')
            plt.plot(range(self.seq_len, self.seq_len + self.pred_len), trend_data, linestyle='--', linewidth=0.5, color='orange')
            plt.plot(range(self.seq_len, self.seq_len + self.pred_len), pred_data, color='red')
            plt.show()
            if self.discrete_var:
                print(f"is high?: {is_high}")
    
    def _visualizing_loss_curve(self):
        plt.plot(self.loss_list)
        plt.show()

    def test(self, label, visualizing_num):
        if label == 'valid':
            dataset = self.valid_dataset
            dataloader = self.valid_loader
        else:
            dataset = self.test_dataset
            dataloader = self.test_loader

        print(f"==== start {label}")
        for model, _ in self.model_list:
            model.eval()
        total_loss = 0
        for x, y in dataloader:
            loss = self._train_one_step(x, y)
            total_loss += loss
        print(f"{label} loss: {np.round(total_loss, 3)}")

        self._visualizing(dataset, visualizing_num)
        