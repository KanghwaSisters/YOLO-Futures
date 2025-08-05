import copy
import time
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
                 path, print_log_interval, print_env_log_interval, save_visual_log=False
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
        self.print_log_interval = print_log_interval
        self.print_env_log_interval = print_env_log_interval
        self.save_visual_log = save_visual_log

        # flag 
        self.n_reset_init_budget = 0
        self.dataset_flag = 0

        # model 
        self.latest_model = None
        self.reward_king_model = None
        self.models_per_steps = deque(maxlen=10)

        # indicator 
        self.pnls = []
        self.durations = []
        self.n_bankruptcys = []
        self.train_iter_rewards_all = {}
        
        # 학습 추적용 변수 추가
        self.train_rewards_history = []
        self.train_losses_history = []

        self.path = path
        self.v_path = path + "/" + "visualization"
        self.m_path = path + "/" + "models"


        ensure_dir(self.path)
        ensure_dir(self.v_path)
        ensure_dir(self.m_path)

        self.log_file = os.path.join(self.path, "train_log.txt")
        with open(self.log_file, "w") as f:
            f.write("==== Training Log Start ====\n")

    def train_visualization(self):
        fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(18, 12))

        fig.suptitle("Train Visualization", fontsize=18)

        plot_training_curves(ax[0], self.train_rewards_history, self.train_losses_history)
        plot_both_pnl_ticks(ax[1], list(range(len(self.pnls))), self.pnls)

        if len(self.durations) != 0:
            plot_maintained_length(ax[2], self.durations)
        else:
            ax[2].axis('off')

        path = self.v_path + '/' + f'TFI{self.dataset_flag}'

        plt.savefig(path)
        self.log(f"✅ 시각화 저장 완료: {path}")

    def __call__(self):

        start_time = time.time()

        for idx, (train_interval, valid_interval) in enumerate(self.train_valid_timestep):
            self.log(f"== [{idx}] interval training ===========================")
            self.dataset_flag = idx

            self.train_env, self.valid_env = self.set_env(train_interval, valid_interval)

            message = f">>>> Train : {train_interval}"
            print(message)
            self.log(message)
            self.train(self.train_env, self.agent)
            self.train_visualization()

            self.log(f">>>> Valid : {valid_interval}")

            models = {'latest model' : self.latest_model, 
                      'highest reward' : self.reward_king_model, 
                      'per steps' : self.models_per_steps[0]}
            
            valid_data = {'timesteps' : self.valid_env.dataset.timesteps,
                          'market' : self.valid_env.dataset.close_prices}

            model_actions_all = {}
            model_pnls_all = {}
            model_volumes_all = {}
            model_rewards_all = {}
            model_r_pnls_all = {}
            model_equities_all = {}  # 새로 추가: 자산 변화 추적
            model_contracts_all = {}

            for key, model in models.items():
                rewards, strengths, assets, r_pnls, actions, equities, contracts = self.valid(self.valid_env, self.valid_agent, key, model)
                model_rewards_all[key] = rewards
                model_volumes_all[key] = strengths
                model_pnls_all[key] = assets
                model_r_pnls_all[key] = r_pnls
                model_actions_all[key] = actions
                model_equities_all[key] = equities  # 새로 추가
                model_contracts_all[key] = contracts # 새로 추가 

            valid_data['model_rewards_all'] = model_rewards_all
            valid_data['model_volumes_all'] = model_volumes_all
            valid_data['model_pnls_all'] = model_pnls_all
            valid_data['model_r_pnls_all'] = model_r_pnls_all
            valid_data['model_actions_all'] = model_actions_all
            valid_data['model_equities_all'] = model_equities_all  # 새로 추가
            valid_data['model_contracts_all'] = model_contracts_all # 새로 추가 

            
            # 학습 데이터 추가
            valid_data['train_rewards'] = self.train_rewards_history
            valid_data['train_losses'] = self.train_losses_history

            self.plot_all_validation_graphs(valid_data, self.v_path)
            self.save_model_to(self.m_path)

        self.time_is(start_time, 'Total')

    def plot_all_validation_graphs(self, valid_data, save_path):
        
        path = save_path + '/' + f'VI{self.dataset_flag}'

        timesteps = valid_data['timesteps']
        market = valid_data['market']
        model_actions_all = valid_data['model_actions_all']
        model_rewards_all = valid_data['model_rewards_all'] 
        model_volumes_all = valid_data['model_volumes_all']
        model_r_pnls_all = valid_data['model_r_pnls_all']
        model_pnls_all = valid_data['model_pnls_all']
        model_equities_all = valid_data['model_equities_all']  # 새로 추가
        model_contracts_all = valid_data['model_contracts_all']
        train_rewards = valid_data['train_rewards']
        train_losses = valid_data['train_losses']
        reset_point = 50

        fig, axs = plt.subplots(14, 1, figsize=(18, 36))
        fig.suptitle("Enhanced Validation Visualization", fontsize=18)

        for idx, (name, actions) in  enumerate(model_actions_all.items()):
            actions = np.array(actions)
            plot_market_with_actions(axs[idx], name, timesteps, market, actions, reset_point, model_contracts_all[name])
        plot_pnls(axs[3], timesteps, model_r_pnls_all, reset_point)
        plot_rewards(axs[4], model_rewards_all)
        plot_volumes(axs[5], model_volumes_all)
        for idx, (name, pnls) in enumerate(model_pnls_all.items()):
            plot_both_pnl_ticks(axs[6+idx], list(range(len(pnls))), pnls)

        # 7. 자산 변화 곡선
        plot_equity_curves(axs[9], timesteps, model_equities_all, reset_point, self.start_budget)
        
        # 8. 포지션 추적 (대표 모델 1개만 - latest model)
        latest_actions = model_actions_all['latest model']
        plot_position_tracking(axs[10], timesteps, latest_actions, reset_point)
        
        # 9. 드로우다운 분석 (리스크 시각화)
        plot_drawdown_analysis(axs[11], timesteps, model_equities_all, reset_point, self.start_budget)
        
        # 10. 학습 곡선 (보상 & 손실)
        # plot_training_curves(axs[12], train_rewards, train_losses)
        
        # 11. 액션 분포 히트맵 (대표 모델의 액션 패턴)
        latest_actions = model_actions_all['latest model']
        plot_action_distribution_heatmap(axs[12], timesteps, latest_actions, self.agent.n_actions)

        for ax in axs.flatten():
            ax.tick_params(axis='x', rotation=45)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(path)
        self.log(f"✅ 시각화 저장 완료 (학습 상태 포함): {path}")
        
        # 메모리 정리
        plt.close(fig)

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
    
    def switch_state(self, env, state):
        return state if env.next_state is None else env.conti()

    def train(self, env, agent):
        
        start_time = time.time()

        episode_rewards = []
        self.train_rewards_history = []
        self.train_losses_history = []

        n_bankruptcy = 0
        max_ep_reward = float('-inf')
        episode = 0

        self.memory = []
        self.durations = []

        # 학습 추적 초기화
        interval_rewards = []
        interval_losses = []

        state = env.reset()

        while not env.dataset.reach_end(env.current_timestep):
            # on-policy의 핵심 : 매 iter마다 메모리 초기화 
            done = False 

            state = self.switch_state(env, state)

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

                self.memory.append([
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

                # 지표 저장 
                self.pnls.append((env.account.unrealized_pnl, env.account.net_realized_pnl))
                
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
            
            # 학습 추적 데이터 기록
            loss = None
            if len(self.memory) >= agent.batch_size:
                advantage = agent.cal_advantage(self.memory)
                loss = agent.train(self.memory, advantage)
                self.memory = []  # 학습 후에만 초기화

            self.train_rewards_history.append(ep_reward)
            self.train_losses_history.append(loss)

            action_prop = (ep_n_positions / sum(ep_n_positions) * 100).round(0)

            if (loss != None) & ((episode+1) % (self.print_log_interval == 0 or env.info in ['done', 'bankrupt', 'end_of_data', 'margin_call', 'insufficient', 'maturity_data'])):
                self.log(f"[{self.dataset_flag}|Train] Ep {episode+1:03d} | info: {env.info} | Maintained for: {env.maintained_steps} | Reward: {ep_reward:4.0f} | Loss: {loss:6.3f} | Pos(short/hold/long): {int(action_prop[-1])}% / {int(action_prop[0])}% / {int(action_prop[1])}% | Strength: {ep_execution_strength / max(ep_len,1):.2f} |")
            
            if (episode+1) % self.print_env_log_interval == 0:
                print(env)
                self.log(env.__str__())

            if env.info == 'bankrupt':
                # print(env)
                self.log(env.__str__())
                self.durations.append(env.maintained_steps)
                n_bankruptcy += 1
                env.account.reset()
                env.performance_tracker.reset()
                env.risk_metrics.reset()
                
                # 시각화 
                if self.save_visual_log:
                    _, ax = plt.subplots(figsize=(12,6))
                    plot_both_pnl_ticks(ax, list(range(len(self.pnls))), self.pnls)
                    plt.tight_layout()

                    path = self.v_path + '/' + f'T{self.dataset_flag}I{n_bankruptcy}'

                    plt.savefig(path)
                    self.log(f"✅ 시각화 저장 완료: {path}")

                self.pnls = []

            episode += 1


        self.n_bankruptcys.append(n_bankruptcy)
        
        # 인터벌별 학습 데이터 저장
        self.train_rewards_history.extend(interval_rewards)
        self.train_losses_history.extend(interval_losses)

        self.log(f"\n== [Train 결과 요약: Interval {self.dataset_flag}] ==============================")
        self.log(f"  - 총 에피소드 수: {episode}")
        self.log(f"  - 최대 보상: {max_ep_reward:.2f}")
        self.log(f"  - 최종 평균 보상: {np.mean(episode_rewards):.2f}")
        self.log(f"  - 파산 횟수: {n_bankruptcy}")
        self.log(f"  - 파산 전 평균 유지 스텝 수: {np.mean(self.durations[-episode:])}")
        self.log(f"  - 모델 저장 간격 (10)")
        self.log("============================================================\n")

        self.time_is(start_time, f'Train:I{self.dataset_flag}')

    def valid(self, env, agent, model_name, state_dict):

        start_time = time.time()

        agent.load_model(state_dict)

        durations = []
        asset_history = []
        env_execution_strengths = []
        episode_rewards = []
        actions = []
        equity_history = []  # 새로 추가: 총 자산 추적
        contract_history = []

        moving_avg_rewards = deque(maxlen=self.ma_interval)
        episode_execution_strengths = deque(maxlen=self.n_steps)
        maintained_steps = 0
        episode = 0
        n_bankruptcy = 0
        pnls = []

        state = env.reset()

        while not env.dataset.reach_end(env.current_timestep):
            done = False
            state = self.switch_state(env, state)

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

                pnls.append((env.account.unrealized_pnl, env.account.net_realized_pnl))
                env_execution_strengths.append(env.account.execution_strength)
                asset_history.append(env.account.realized_pnl)
                actions.append(action)
                contract_history.append(env.account.current_position * env.account.execution_strength)
                
                # 새로 추가: 총 자산 (available_balance + unrealized_pnl) 기록
                current_equity = env.account.available_balance + env.account.unrealized_pnl
                equity_history.append(max(current_equity, 1.0))  # 음수 방지


            # 에피소드 종료 후 기록
            maintained_steps += ep_len
            episode_rewards.append(ep_reward)
            moving_avg_rewards.append(ep_reward)
            episode_execution_strengths.append(ep_execution_strength)

            action_prop = (ep_n_positions / sum(ep_n_positions) * 100).round(0)

            if env.info in ['bankrupt']:
                durations.append(maintained_steps)
                n_bankruptcy += 1
                maintained_steps = 0
                # 초기화 
                env.account.reset()
                env.performance_tracker.reset()
                env.risk_metrics.reset()

                pnls = []
                contract_history = []


            self.log(f"[{self.dataset_flag}|Valid] Ep {episode+1:03d} | info: {env.info} | Maintained for: {maintained_steps} | Reward: {ep_reward:4.0f} | Pos(short/hold/long): {int(action_prop[-1])}% / {int(action_prop[0])}% / {int(action_prop[1])}% | Strength: {ep_execution_strength / max(ep_len,1):.2f} |")

            if (episode+1) % self.print_env_log_interval == 0:
                print(env)
                self.log(env.__str__())

            episode += 1

        self.log(f"\n==[Valid 결과 요약:Interval{self.dataset_flag},{model_name}] ==============================")
        self.log(f"  - 총 에피소드 수: {episode}")
        self.log(f"  - 평균 보상: {np.mean(episode_rewards):.2f}")
        self.log(f"  - 마지막 수익: {asset_history[-1]:.2f}")
        self.log(f"  - 최종 총 자산: {equity_history[-1]:.2f}")  # 새로 추가
        self.log("============================================================\n")

        self.time_is(start_time, f'Valid:{model_name}')

        return episode_rewards, env_execution_strengths, pnls, asset_history, actions,equity_history, contract_history

    def save_model_to(self, path):
        # 가장 최근 모델 저장
        if self.latest_model is not None:
            torch.save(self.latest_model, os.path.join(path, f'I{self.dataset_flag+1}latest.pth'))
            self.log("[Saved] latest_model")

        # 최고 보상 모델 저장
        if self.reward_king_model is not None:
            torch.save(self.reward_king_model, os.path.join(path, f'I{self.dataset_flag+1}bestreward.pth'))
            self.log("[Saved] reward_king_model")

        # n-step마다 모델 저장 
        recent_models = list(self.models_per_steps)
        for idx, model_state in enumerate(recent_models):
            torch.save(model_state, os.path.join(path, f'I{self.dataset_flag+1}_{(idx+1)}steps.pth'))
        self.log(f"[Saved] {len(recent_models)} recent models")

    def save(self, CONFIG):

        # 설정 문자열 생성
        config_lines = []
        config_lines.append("===== CONFIGURATION SETTINGS =====\n")
        for key, value in CONFIG.items():
            config_lines.append(f"{key}: {value}")
        config_str = "\n".join(config_lines)

        save_path = os.path.join(self.path, "setting.txt")

        # 파일로 저장
        with open(save_path, "w") as f:
            f.write(config_str)

        self.log(f"✅ 설정 저장 완료: {save_path}")


    def split_position_strength(self, action):
        if action == 0:
            return 0, 0
        
        strength = np.abs(action)
        position = np.sign(action)
        return position.item(), strength.item()
    
    def log(self, message):
        with open(self.log_file, "a") as f:
            f.write(message + "\n")

    def time_is(self, start_time, status):
        elapsed_time = time.time() - start_time
        hours, rem = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(rem, 60)

        message = f"⏱️ {status} 시간 소요: {int(hours)}시간 {int(minutes)}분 {int(seconds)}초\n"

        print(message)
        self.log(message)

class HorizonBoundNonEpisodicTrainer(NonEpisodicTrainer):
    def __init__(self, df, env, train_valid_timestep, window_size, state, reward_ftn, done_ftn, start_budget, scaler, position_cap, # env 관련 파라미터 
                 agent, model, optimizer, device,  # agent 관련 파라미터 
                 n_steps, ma_interval, save_interval,
                 path, print_log_interval, print_env_log_interval, save_visual_log=False
                 ):
        super().__init__(df, env, train_valid_timestep, window_size, state, reward_ftn, done_ftn, start_budget, scaler, position_cap, # env 관련 파라미터 
                        agent, model, optimizer, device,  
                        n_steps, ma_interval, save_interval,
                        path, print_log_interval, print_env_log_interval, save_visual_log)
        self.episode_pnl = []

    def switch_state(self, env, state):
        if env.next_state == None:
            return state
        else:
            return env.conti()

    def train(self, env, agent):
        
        start_time = time.time()

        episode_rewards = []
        episode_execution_strengths = deque(maxlen=self.n_steps)

        n_bankruptcy = 0
        max_ep_reward = float('-inf')
        episode = 0

        self.memory = []
        self.durations = []

        # 학습 추적 초기화
        interval_rewards = []
        interval_losses = []

        state = env.reset()

        while not env.dataset.reach_end(env.current_timestep):
            # on-policy의 핵심 : 매 iter마다 메모리 초기화 
            done = False 

            state = self.switch_state(env, state)

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

                self.memory.append([
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

                # 지표 저장 
                self.pnls.append((env.account.unrealized_pnl, env.account.net_realized_pnl))

            # save model dicts 
            if max_ep_reward <= ep_reward:
                self.reward_king_model = agent.model.state_dict()
                max_ep_reward = ep_reward 

            if episode % self.save_interval == 0:
                self.models_per_steps.append(agent.model.state_dict())

            self.latest_model = agent.model.state_dict()
                
            episode_rewards.append(ep_reward)
            episode_execution_strengths.append(ep_execution_strength)
            
            # 학습 추적 데이터 기록
            loss = None
            if len(self.memory) >= agent.batch_size:
                advantage = agent.cal_advantage(self.memory)
                loss = agent.train(self.memory, advantage)
                self.memory = []  # 학습 후에만 초기화

            action_prop = (ep_n_positions / sum(ep_n_positions) * 100).round(0)

            if (loss != None) & (((episode+1) % self.print_log_interval == 0) or env.info in ['done', 'bankrupt', 'end_of_data', 'margin_call', 'insufficient', 'maturity_data']):
                self.log(f"[{self.dataset_flag}|Train] Ep {episode+1:03d} | info: {env.info} | Maintained for: {env.maintained_steps} | Reward: {ep_reward:4.0f} | Loss: {loss:6.3f} | Pos(short/hold/long): {int(action_prop[-1])}% / {int(action_prop[0])}% / {int(action_prop[1])}% | Strength: {ep_execution_strength / max(ep_len,1):.2f} |")
            
            if (episode+1) % self.print_env_log_interval == 0:
                print(env)
                self.log(env.__str__())

            if env.info == 'bankrupt':
                self.log(env.__str__())
                self.durations.append(env.maintained_steps)
                n_bankruptcy += 1
                self.episode_pnl.append((env.account.available_balance - env.account.initial_budget)/env.account.initial_budget)

                # reset 
                env.maintained_steps = 0
                env.account.reset()
                env.performance_tracker.reset()
                env.risk_metrics.reset()
                
                # 시각화 
                if self.save_visual_log:
                    _, ax = plt.subplots(figsize=(12,6))
                    plot_both_pnl_ticks(ax, list(range(len(self.pnls))), self.pnls)
                    plt.tight_layout()

                    path = self.v_path + '/' + f'T{self.dataset_flag}I{n_bankruptcy}'

                    plt.savefig(path)
                    self.log(f"✅ 시각화 저장 완료: {path}")

                self.pnls = []

            elif env.info == 'done':
                self.log(env.__str__())
                self.durations.append(env.maintained_steps)
                self.episode_pnl.append((env.account.available_balance - env.account.initial_budget)/env.account.initial_budget)

                env.maintained_steps = 0
                env.account.reset()
                env.performance_tracker.reset()
                env.risk_metrics.reset()

            episode += 1

        self.n_bankruptcys.append(n_bankruptcy)
        
        # 인터벌별 학습 데이터 저장
        self.train_rewards_history.extend(interval_rewards)
        self.train_losses_history.extend(interval_losses)

        self.log(f"\n== [Train 결과 요약: Interval {self.dataset_flag}] ==============================")
        self.log(f"  - 총 에피소드 수: {episode}")
        self.log(f"  - 최대 보상: {max_ep_reward:.2f}")
        self.log(f"  - 최종 평균 보상: {np.mean(episode_rewards):.2f}")
        self.log(f"  - 파산 횟수: {n_bankruptcy}")
        self.log(f"  - 파산 전 평균 유지 스텝 수: {np.mean(self.durations[-episode:])}")
        self.log(f"  - {env.max_step} 에피소드 별 수익률: {np.mean(self.episode_pnl).item() * 100:.2f} %")
        self.log(f"  - 모델 저장 간격 (10)")
        self.log("============================================================\n")

        self.time_is(start_time, f'Train:I{self.dataset_flag}')

    def valid(self, env, agent, model_name, state_dict):

        start_time = time.time()

        agent.load_model(state_dict)

        episode_pnl = []
        durations = []
        asset_history = []
        env_execution_strengths = []
        episode_rewards = []
        actions = []
        equity_history = []  # 새로 추가: 총 자산 추적
        contract_history = []

        moving_avg_rewards = deque(maxlen=self.ma_interval)
        episode_execution_strengths = deque(maxlen=self.n_steps)
        maintained_steps = 0
        episode = 0
        n_bankruptcy = 0
        pnls = []

        state = env.reset()

        while not env.dataset.reach_end(env.current_timestep):
            done = False
            state = self.switch_state(env, state)

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

                pnls.append((env.account.unrealized_pnl, env.account.net_realized_pnl))
                env_execution_strengths.append(env.account.execution_strength)
                asset_history.append(env.account.realized_pnl)
                actions.append(action)
                contract_history.append(env.account.current_position * env.account.execution_strength)
                
                # 새로 추가: 총 자산 (available_balance + unrealized_pnl) 기록
                current_equity = env.account.available_balance + env.account.unrealized_pnl
                equity_history.append(max(current_equity, 1.0))  # 음수 방지

            # 에피소드 종료 후 기록
            maintained_steps += ep_len
            episode_rewards.append(ep_reward)
            moving_avg_rewards.append(ep_reward)
            episode_execution_strengths.append(ep_execution_strength)

            action_prop = (ep_n_positions / sum(ep_n_positions) * 100).round(0)

            if env.info == 'bankrupt':
                durations.append(maintained_steps)
                n_bankruptcy += 1
                maintained_steps = 0
                episode_pnl.append((env.account.available_balance - env.account.initial_budget)/env.account.initial_budget)
                
                # 초기화 
                env.account.reset()
                env.performance_tracker.reset()
                env.risk_metrics.reset()

                pnls = []
                contract_history = []
            
            elif env.info == 'done':
                durations.append(maintained_steps)
                episode_pnl.append((env.account.available_balance - env.account.initial_budget)/env.account.initial_budget)

                maintained_steps = 0
                env.account.reset()
                env.performance_tracker.reset()
                env.risk_metrics.reset()

            self.log(f"[{self.dataset_flag}|Valid] Ep {episode+1:03d} | info: {env.info} | Maintained for: {maintained_steps} | Reward: {ep_reward:4.0f} | Pos(short/hold/long): {int(action_prop[-1])}% / {int(action_prop[0])}% / {int(action_prop[1])}% | Strength: {ep_execution_strength / max(ep_len,1):.2f} |")

            if (episode+1) % self.print_env_log_interval == 0:
                print(env)
                self.log(env.__str__())

            episode += 1

        self.log(f"\n==[Valid 결과 요약:Interval{self.dataset_flag},{model_name}] ==============================")
        self.log(f"  - 총 에피소드 수: {episode}")
        self.log(f"  - 평균 보상: {np.mean(episode_rewards):.2f}")
        self.log(f"  - 마지막 수익: {asset_history[-1]:.2f}")
        self.log(f"  - 최종 총 자산: {equity_history[-1]:.2f}")  # 새로 추가
        self.log(f"  - {env.max_step} 에피소드 별 수익률: {np.mean(episode_pnl).item() * 100:.2f} %")
        self.log("============================================================\n")

        self.time_is(start_time, f'Valid:{model_name}')

        return episode_rewards, env_execution_strengths, pnls, asset_history, actions,equity_history, contract_history
