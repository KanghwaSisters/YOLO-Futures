import copy
import torch
from collections import deque
import matplotlib.pyplot as plt
import numpy as np
import os

from utils.ensureDir import *
from visualization.methods import *
from trainer.nonEpisodic import *


class EpisodicTrainer(NonEpisodicTrainer):
    def __init__(self, df, env, train_valid_timestep, window_size, state, reward_ftn, done_ftn, start_budget, scaler, position_cap, # env 관련 파라미터 
                 agent, model, optimizer, device,  # agent 관련 파라미터 
                 n_steps, ma_interval, save_interval,
                 path,print_log_interval, print_env_log_interval,
                 max_iter_same_interval=200
                 ):
        
        super().__init__(df, env, train_valid_timestep, window_size, state, reward_ftn, done_ftn, start_budget, scaler, position_cap, # env 관련 파라미터 
                            agent, model, optimizer, device,  # agent 관련 파라미터 
                            n_steps, ma_interval, save_interval,
                            path, print_log_interval, print_env_log_interval)
        
        # EpisodicTrainer 고유 
        self.max_iter_same_interval = max_iter_same_interval
        self.remaining_n = self.max_iter_same_interval
        self.prev_dataset_flag = -1

    def switch_state(self, env, state):
        if self.prev_dataset_flag != self.dataset_flag:
             # 데이터 셋이 바뀌면 remaining_n을 업데이트한다. 
             self.prev_dataset_flag = self.dataset_flag
             self.remaining_n = self.max_iter_same_interval

        if self.remaining_n > 0 and env.info in ['bankrupt']:
                self.remaining_n -= 1
                self.log(f">>>>> reset the env : {env.info} occured. Go Back To Start.")
                return env.reset()
            
        elif env.next_state == None:
            return state
        else:
            return env.conti()
        

class HorizonBoundEpisodicTrainer(HorizonBoundNonEpisodicTrainer):
    def __init__(self, df, env, train_valid_timestep, window_size, state, reward_ftn, done_ftn, start_budget, scaler, position_cap, # env 관련 파라미터 
                 agent, model, optimizer, device,  # agent 관련 파라미터 
                 n_steps, ma_interval, save_interval,
                 path, print_log_interval, print_env_log_interval, save_visual_log=False,
                 max_iter_same_interval=1000
                 ):
        super().__init__(df, env, train_valid_timestep, window_size, state, reward_ftn, done_ftn, start_budget, scaler, position_cap, # env 관련 파라미터 
                        agent, model, optimizer, device,  
                        n_steps, ma_interval, save_interval,
                        path, print_log_interval, print_env_log_interval, save_visual_log)
        
    # EpisodicTrainer 고유 
        self.max_iter_same_interval = max_iter_same_interval
        self.remaining_n = self.max_iter_same_interval
        self.prev_dataset_flag = -1

    def switch_state(self, env, state):
        if self.prev_dataset_flag != self.dataset_flag:
             # 데이터 셋이 바뀌면 remaining_n을 업데이트한다. 
             self.prev_dataset_flag = self.dataset_flag
             self.remaining_n = self.max_iter_same_interval

        if self.remaining_n > 0 and env.info in ['bankrupt']:
                self.remaining_n -= 1
                self.log(f">>>>> reset the env : {env.info} occured. Go Back To Start.")
                return env.reset()
        
        # 처음 시작 
        elif env.next_state == None:
            return state
        
        else:
            return env.conti()