import copy
import torch
from collections import deque  
import matplotlib.pyplot as plt
import numpy as np
import os

from utils.ensureDir import *
from visualization.methods import *


class NonEpisodicTrainer:
    def __init__(self, df, env, train_valid_timestep, window_size, state, reward_ftn, done_ftn, start_budget, scaler, position_cap, # env 관련 파라미터 
                 agent, model, optimizer, device,  # agent 관련 파라미터 
                 n_steps, ma_interval, save_interval,
                 path
                 ):
        
        # dataframe 
        self.df = df 

        # env 관련 파라미터 
        self.env = env
        self.train_valid_timestep = train_valid_timestep
        self.window_size = window_size
        self.state = state
        self.reward_ftn = reward_ftn
        self.done_ftn = done_ftn
        self.start_budget = start_budget
        self.scaler = scaler
        self.position_cap = position_cap

        # agent 관련 파라미터 
        self.agent = agent
        self.agent.set_optimizer(optimizer)
        self.valid_agent = copy.deepcopy(self.agent)
        self.model = model

        # etc
        self.device = device
        self.n_steps = n_steps
        self.ma_interval = ma_interval
        self.save_interval = save_interval

        # flag 
        self.n_reset_init_budget = 0
        self.dataset_flag = 0

        # model 
        self.latest_model = None
        self.reward_king_model = None
        self.models_per_steps = deque(maxlen=10)

        # indicator 
        self.durations = []
        self.n_bankruptcys = []

        self.path = path
    
    def __call__(self, ):
        for idx, (train_interval, valid_interval) in enumerate(self.train_valid_timestep):
            print(f"== [{idx}] interval training ===========================")
            self.dataset_flag = idx

            self.train_env, self.valid_env = self.set_env(train_interval, valid_interval)

            print(f">>>> Train : {train_interval}")
            self.train(self.train_env, self.agent)

            print(f">>>> Valid : {valid_interval}")

            models = {'latest model' : self.latest_model, 
                      'highest reward' : self.reward_king_model, 
                      'per steps' : self.models_per_steps[0]}
            
            valid_data = {'timesteps' : self.valid_env.dataset.timesteps,
                          'market' : self.valid_env.dataset.close_prices}

            model_actions_all = {}
            model_pnls_all = {}
            model_volumes_all = {}
            model_rewards_all = {}
            model_durations_all = {}

            for key, model in models.items():
                rewards, strengths, assets, durations, actions = self.valid(self.valid_env, self.valid_agent, key, model)
                model_rewards_all[key] = rewards
                model_volumes_all[key] = strengths
                model_pnls_all[key] = assets
                model_durations_all[key] = durations
                model_actions_all[key] = actions

            valid_data['model_rewards_all'] = model_rewards_all
            valid_data['model_volumes_all'] = model_volumes_all
            valid_data['model_pnls_all'] = model_pnls_all
            valid_data['model_durations_all'] = model_durations_all
            valid_data['model_actions_all'] = model_actions_all

            self.plot_all_validation_graphs(valid_data, self.path)
            self.save_model_to(self.path)

    def plot_all_validation_graphs(self, valid_data, save_path):
        
        path = save_path + '/' + f'I{self.dataset_flag}V'

        timesteps = valid_data['timesteps']
        market = valid_data['market']
        model_actions_all = valid_data['model_actions_all']
        model_rewards_all = valid_data['model_rewards_all'] 
        model_volumes_all = valid_data['model_volumes_all']
        model_pnls_all = valid_data['model_pnls_all']
        model_durations_all = valid_data['model_durations_all']
        reset_point = 50


        fig, axs = plt.subplots(7, 1, figsize=(18, 21))
        fig.suptitle("Validation Visualization", fontsize=18)

        for idx, (name, actions) in  enumerate(model_actions_all.items()):
            actions = np.array(actions)
            plot_market_with_actions(axs[idx], name, timesteps, market, actions, reset_point)
        plot_pnls(axs[3], timesteps, model_pnls_all, reset_point)
        plot_rewards(axs[4], model_rewards_all)
        plot_volumes(axs[5], model_volumes_all)
        plot_durations(axs[6], model_durations_all)

        # axs[3, 1].axis('off')

        for ax in axs.flatten():
            ax.tick_params(axis='x', rotation=45)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(path)
        print(f"✅ 시각화 저장 완료: {path}")

    def set_env(self, time_interval_train:tuple, time_interval_valid:tuple):

        train_env = self.env(full_df=self.df, 
                                        date_range=time_interval_train, 
                                        window_size=self.window_size, 
                                        state_type=self.state, 
                                        reward_ftn=self.reward_ftn, 
                                        done_ftn=self.done_ftn, 
                                        start_budget=self.start_budget,
                                        n_actions=self.agent.n_actions,
                                        scaler=self.scaler,
                                        position_cap=self.position_cap)
        
        valid_env = self.env(full_df=self.df, 
                                        date_range=time_interval_valid, 
                                        window_size=self.window_size, 
                                        state_type=self.state, 
                                        reward_ftn=self.reward_ftn, 
                                        done_ftn=self.done_ftn, 
                                        start_budget=self.start_budget,
                                        n_actions=self.agent.n_actions,
                                        scaler=self.scaler,
                                        position_cap=self.position_cap)
        
        return train_env, valid_env
        

    def train(self, env, agent):
        episode_rewards = []
        moving_avg_rewards = deque(maxlen=self.ma_interval)
        episode_execution_strengths = deque(maxlen=self.n_steps)

        n_bankruptcy = 0
        max_ep_reward = float('-inf')
        episode = 0

        maintained_steps = 0
        memory = []

        state = env.reset()

        while not env.dataset.reach_end(env.current_timestep):
            # on-policy의 핵심 : 매 iter마다 메모리 초기화 
            done = False 

            state = state if env.next_state is None else env.conti()

            if type(state) == tuple:
                ts_state = torch.tensor(state[0], dtype=torch.float32).unsqueeze(0).to(self.device)
                agent_state = torch.tensor(state[1], dtype=torch.float32).unsqueeze(0).to(self.device)

                state = (ts_state, agent_state)
            
            ep_reward = 0
            ep_len = 0
            ep_n_positions = np.array([0, 0, 0]) # 순서대로 0, 1, -1
            ep_execution_strength = 0

            for _ in range(self.n_steps):
                if done:
                    break
                mask = env.mask

                action, log_prob = agent.get_action(state, mask)
                next_state, reward, done = env.step(action)
                current_position, execution_strength = self.split_position_strength(action)

                if type(state) == tuple:
                    ts_state = torch.tensor(next_state[0], dtype=torch.float32).unsqueeze(0).to(self.device)
                    agent_state = torch.tensor(next_state[1], dtype=torch.float32).unsqueeze(0).to(self.device)
                    next_state = (ts_state, agent_state)

                memory.append([
                    state,
                    torch.tensor([[action]]),
                    torch.tensor([reward], dtype=torch.float32),
                    next_state,
                    torch.tensor([done], dtype=torch.float32),
                    torch.tensor([log_prob], dtype=torch.float32),
                    torch.tensor([mask], dtype=torch.bool)
                ])

                # update step 지표 
                state = next_state
                ep_reward += reward
                ep_len += 1
                ep_n_positions[current_position] += 1
                ep_execution_strength += execution_strength

            # update
            maintained_steps += ep_len

            # save model dicts 
            if max_ep_reward <= ep_reward:
                self.reward_king_model = agent.model.state_dict()
                max_ep_reward = ep_reward 

            if episode % self.save_interval == 0:
                self.models_per_steps.append(agent.model.state_dict())

            self.latest_model = agent.model.state_dict()
                
            episode_rewards.append(ep_reward)
            moving_avg_rewards.append(ep_reward)
            episode_execution_strengths.append(ep_execution_strength)

            loss = None
            if len(memory) >= agent.batch_size:
                advantage = agent.cal_advantage(memory)
                loss = agent.train(memory, advantage)
                memory = []  # 학습 후에만 초기화

            action_prop = (ep_n_positions / sum(ep_n_positions) * 100).round(0)

            if loss != None:
                print(f"[{self.dataset_flag}|Train] Ep {episode+1:03d} | info: {env.info} | Maintained for: {maintained_steps} | Reward: {ep_reward:4.0f} | Loss: {loss:6.3f} | Pos(short/hold/long): {int(action_prop[-1])}% / {int(action_prop[0])}% / {int(action_prop[1])}% | Strength: {ep_execution_strength / max(ep_len,1):.2f} |")
            
            if (episode+1) % 50 == 0:
                print(env)

            # 지표 저장 
            if env.info in ['margin_call', 'maturity_data', 'bankrupt']:
                self.durations.append(maintained_steps)
                n_bankruptcy += 1
                maintained_steps = 0
                env.account.reset()

            episode += 1

        self.n_bankruptcys.append(n_bankruptcy)

        print(f"\n== [Train 결과 요약: Interval {self.dataset_flag}] ==============================")
        print(f"  - 총 에피소드 수: {episode}")
        print(f"  - 최대 보상: {max_ep_reward:.2f}")
        print(f"  - 최종 평균 보상: {np.mean(episode_rewards):.2f}")
        print(f"  - 파산 횟수: {n_bankruptcy}")
        print(f"  - 파산 전 평균 유지 스텝 수: {np.mean(self.durations[-episode:])}")
        print(f"  - 모델 저장 간격 (10)")
        print("============================================================\n")

    def valid(self, env, agent, model_name, state_dict):

        agent.load_model(state_dict)

        durations = []
        asset_history = []
        env_execution_strengths = []
        episode_rewards = []
        actions = []

        moving_avg_rewards = deque(maxlen=self.ma_interval)
        episode_execution_strengths = deque(maxlen=self.n_steps)
        maintained_steps = 0
        episode = 0
        n_bankruptcy = 0

        state = env.reset()

        while not env.dataset.reach_end(env.current_timestep):
            done = False
            state = state if env.next_state is None else env.conti()

            if type(state) == tuple:
                ts_state = torch.tensor(state[0], dtype=torch.float32).unsqueeze(0).to(self.device)
                agent_state = torch.tensor(state[1], dtype=torch.float32).unsqueeze(0).to(self.device)
                state = (ts_state, agent_state)

            ep_reward = 0
            ep_len = 0
            ep_n_positions = np.array([0, 0, 0])  # hold, long, short
            ep_execution_strength = 0

            for _ in range(self.n_steps):
                if done:
                    break
                mask = env.mask

                action, _ = agent.get_action(state, mask)
                next_state, reward, done = env.step(action)
                current_position, execution_strength = self.split_position_strength(action)

                if type(state) == tuple:
                    ts_state = torch.tensor(next_state[0], dtype=torch.float32).unsqueeze(0).to(self.device)
                    agent_state = torch.tensor(next_state[1], dtype=torch.float32).unsqueeze(0).to(self.device)
                    next_state = (ts_state, agent_state)

                state = next_state
                ep_reward += reward
                ep_len += 1
                ep_n_positions[current_position] += 1
                ep_execution_strength += execution_strength

                env_execution_strengths.append(env.account.execution_strength)
                asset_history.append(env.account.realized_pnl)
                actions.append(action)

            # 에피소드 종료 후 기록
            maintained_steps += ep_len
            episode_rewards.append(ep_reward)
            moving_avg_rewards.append(ep_reward)
            episode_execution_strengths.append(ep_execution_strength)

            avg_reward = np.mean(moving_avg_rewards)
            action_prop = (ep_n_positions / sum(ep_n_positions) * 100).round(0)

            if env.info in ['margin_call', 'maturity_data', 'bankrupt']:
                durations.append(maintained_steps)
                n_bankruptcy += 1
                maintained_steps = 0
                env.account.reset()

            print(f"[{self.dataset_flag}|Valid] Ep {episode+1:03d} | info: {env.info} | Maintained for: {maintained_steps} | Reward: {ep_reward:4.0f} | Pos(short/hold/long): {int(action_prop[-1])}% / {int(action_prop[0])}% / {int(action_prop[1])}% | Strength: {ep_execution_strength / max(ep_len,1):.2f} |")

            if (episode+1) % 50 == 0:
                print(env)

            episode += 1

        print(f"\n==[Valid2 결과 요약:Interval{self.dataset_flag}] ==============================")
        print(f"  - 총 에피소드 수: {episode}")
        print(f"  - 평균 보상: {np.mean(episode_rewards):.2f}")
        print(f"  - 평균 유지 시간: {np.mean(durations):.2f} step")
        print(f"  - 마지막 수익: {asset_history[-1]:.2f}")
        print("============================================================\n")

        return episode_rewards, env_execution_strengths, asset_history, durations, actions

    def save_model_to(self, path):
        # 디렉토리 확인 및 생성
        ensure_dir(path)

        # 가장 최근 모델 저장
        if self.latest_model is not None:
            torch.save(self.latest_model, os.path.join(path, f'I{self.dataset_flag+1}latest.pth'))
            print("[Saved] latest_model")

        # 최고 보상 모델 저장
        if self.reward_king_model is not None:
            torch.save(self.reward_king_model, os.path.join(path, f'I{self.dataset_flag+1}bestreward.pth'))
            print("[Saved] reward_king_model")

        # n-step마다 모델 저장 
        recent_models = list(self.models_per_steps)
        for idx, model_state in enumerate(recent_models):
            torch.save(model_state, os.path.join(path, f'I{self.dataset_flag+1}_{(idx+1)}steps.pth'))
        print(f"[Saved] {len(recent_models)} recent models")

    def split_position_strength(self, action):
        if action == 0:
            return 0, 0
        
        strength = np.abs(action)
        position = np.sign(action)
        return position.item(), strength.item()