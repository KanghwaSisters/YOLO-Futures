from trainer.nonEpisodic import *
from env.GoalOrTimeoutEnv import *

class GOTNonEpisodicTrainer(NonEpisodicTrainer):
    def __init__(self, df, env, train_valid_timestep, window_size, state, reward_ftn, done_ftn, start_budget, scaler, position_cap, # env 관련 파라미터 
                 agent, model, optimizer, device,  # agent 관련 파라미터 
                 n_steps, ma_interval, save_interval,
                 path, print_log_interval, print_env_log_interval, save_visual_log=False
                 ):
        super().__init__(df, env, train_valid_timestep, window_size, state, reward_ftn, done_ftn, start_budget, scaler, position_cap, # env 관련 파라미터 
                        agent, model, optimizer, device,  
                        n_steps, ma_interval, save_interval,
                        path, print_log_interval, print_env_log_interval, save_visual_log)
    
    def train(self, env, agent):
        
        start_time = time.time()

        episode_rewards = []
        realized_pnl_ratios = []

        n_bankruptcy = 0
        max_ep_reward = float('-inf')
        episode = 0

        self.memory = []
        self.durations = []
        self.total_pnl = []

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
            
            # update 
            episode_rewards.append(ep_reward)
            realized_pnl_ratios.append(env.account.realized_pnl_ratio)

            
            # 학습 추적 데이터 기록
            loss = None
            if len(self.memory) >= agent.batch_size:
                advantage = agent.cal_advantage(self.memory)
                loss = agent.train(self.memory, advantage)
                self.memory = []  # 학습 후에만 초기화

            action_prop = (ep_n_positions / sum(ep_n_positions) * 100).round(0)

            # update 
            self.train_rewards_history.append(ep_reward)
            self.train_losses_history.append(loss)

            if (loss != None) & ((episode+1) % self.print_log_interval == 0):
                self.log(f"[{self.dataset_flag}|Train] Ep {episode+1:03d} | info: {env.info} | Maintained for: {env.maintained_steps} | PnL Ratio: {env.account.realized_pnl_ratio * 100:.2f} % | Reward: {ep_reward:4.0f} | Loss: {loss:6.3f} | Pos(short/hold/long): {int(action_prop[-1])}% / {int(action_prop[0])}% / {int(action_prop[1])}% | Strength: {ep_execution_strength / max(ep_len,1):.2f} |")
            
            if (episode+1) % self.print_env_log_interval == 0:
                print(env)
                self.log(env.__str__())

            if env.info == 'bankrupt':
                n_bankruptcy += 1
                
                # 시각화 
                if self.save_visual_log:
                    _, ax = plt.subplots(figsize=(12,6))
                    plot_both_pnl_ticks(ax, list(range(len(self.pnls))), self.pnls)
                    plt.tight_layout()

                    path = self.v_path + '/' + f'T{self.dataset_flag}I{n_bankruptcy}'

                    plt.savefig(path)
                    self.log(f"✅ 시각화 저장 완료: {path}")

            if env.info in env.done_status_list:
                self.durations.append(env.maintained_steps)
                self.total_pnl.append(env.account.realized_pnl)

                env.maintained_steps = 0
                env.account.reset()
                env.performance_tracker.reset()
                env.risk_metrics.reset()
                if env.info !='end_of_data':
                    self.pnls = []

            episode += 1

        self.n_bankruptcys.append(n_bankruptcy)
        
        # 인터벌별 학습 데이터 저장
        self.train_rewards_history.extend(interval_rewards)
        self.train_losses_history.extend(interval_losses)

        message = ''
        message += f"\n== [Train 결과 요약: Interval {self.dataset_flag}] ==============================\n"
        message += f"  - 총 에피소드 수         : {episode}\n"
        message += f"  - 최대 보상             : {max_ep_reward:.2f}\n"
        message += f"  - 최종 평균 보상         : {np.mean(episode_rewards):.2f}\n"
        message += f"  - 파산 횟수             : {n_bankruptcy}\n"
        message += f"  - 파산 전 평균 유지 스텝 수: {np.mean(self.durations)}\n"
        message += f"  - 최종 평균 수익률        : {np.mean(realized_pnl_ratios).item() * 100:.2f} %\n"
        message += f"  - 총 실현 수익           : {sum(self.total_pnl)}\n"
        message += "============================================================\n"

        print(message)
        self.log(message)

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
        pnl_ratio_history = []
        total_pnl = []

        moving_avg_rewards = deque(maxlen=self.ma_interval)
        episode_execution_strengths = deque(maxlen=self.n_steps)
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
            episode_rewards.append(ep_reward)
            moving_avg_rewards.append(ep_reward)
            episode_execution_strengths.append(ep_execution_strength)
            pnl_ratio_history.append(env.account.realized_pnl_ratio)

            action_prop = (ep_n_positions / sum(ep_n_positions) * 100).round(0)

            self.log(f"[{self.dataset_flag}|Valid] Ep {episode+1:03d} | info: {env.info} | Maintained for: {env.maintained_steps} | PnL Ratio: {env.account.realized_pnl_ratio * 100:.2f} % | Reward: {ep_reward:4.0f} | Pos(short/hold/long): {int(action_prop[-1])}% / {int(action_prop[0])}% / {int(action_prop[1])}% | Strength: {ep_execution_strength / max(ep_len,1):.2f} |")
            
            if env.info == 'bankrupt':
                n_bankruptcy += 1

            if env.info in env.done_status_list:
                total_pnl.append(env.account.realized_pnl)
                durations.append(env.maintained_steps)
                env.maintained_steps = 0

                env.account.reset()
                env.performance_tracker.reset()
                env.risk_metrics.reset()
                
                if env.info !='end_of_data':
                    pnls = []
                    contract_history = []


            if (episode+1) % self.print_env_log_interval == 0:
                print(env)
                self.log(env.__str__())

            episode += 1

        message = ''
        message += f"\n==[Valid 결과 요약:Interval{self.dataset_flag},{model_name}] ==============================\n"
        message += f"  - 총 에피소드 수  : {episode}\n"
        message += f"  - 평균 보상      : {np.mean(episode_rewards):.2f}\n"
        message += f"  - 마지막 수익    : {asset_history[-1]:.2f}\n"
        message += f"  - 평균 수익비율   : {np.mean(pnl_ratio_history).item() * 100:.2f}\n"
        message += f"  - 실현 손익 총합 : {sum(total_pnl)}\n"
        message += "============================================================\n"

        print(message)
        self.log(message)

        self.time_is(start_time, f'Valid:{model_name}')

        return episode_rewards, env_execution_strengths, pnls, asset_history, actions, equity_history, contract_history


class GOTRandomTrainer(GOTNonEpisodicTrainer):
    def __init__(self, df, env, train_valid_timestep, window_size, state, reward_ftn, done_ftn, start_budget, scaler, position_cap, # env 관련 파라미터 
                 agent, model, optimizer, device,  # agent 관련 파라미터 
                 n_steps, ma_interval, save_interval,
                 path, print_log_interval, 
                 print_env_log_interval, save_visual_log=False, n_iteration=5_000
                 ):
        super().__init__(df, env, train_valid_timestep, window_size, state, reward_ftn, done_ftn, start_budget, scaler, position_cap, # env 관련 파라미터 
                        agent, model, optimizer, device,  
                        n_steps, ma_interval, save_interval,
                        path, print_log_interval, print_env_log_interval, save_visual_log)
    
        self.n_iteration = n_iteration 
        self.TrainEnv = env # GOTRandomEnv
        self.ValidEnv = GoalOrTimeoutEnv

    def set_env(self, time_interval_train:tuple, time_interval_valid:tuple):

        train_env = self.TrainEnv(full_df=self.df, 
                                date_range=time_interval_train, 
                                window_size=self.window_size, 
                                state_type=self.state, 
                                reward_ftn=self.reward_ftn, 
                                done_ftn=self.done_ftn, 
                                start_budget=self.start_budget,
                                n_actions=self.agent.n_actions,
                                scaler=self.scaler,
                                position_cap=self.position_cap)
        
        valid_env = self.ValidEnv(full_df=self.df, 
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
        
        start_time = time.time()

        episode_rewards = []
        realized_pnl_ratios = []

        n_bankruptcy = 0
        max_ep_reward = float('-inf')

        self.memory = []
        self.durations = []
        self.total_pnl = []

        # 학습 추적 초기화
        interval_rewards = []
        interval_losses = []

        state = env.reset()
        done = False 

        for episode in range(self.n_iteration):

            if type(state) == tuple:
                ts_state, agent_state = state
                if ts_state.ndim == 2:
                    ts_state = torch.tensor(ts_state, dtype=torch.float32).unsqueeze(0).to(self.device)
                if agent_state.ndim == 1:
                    agent_state = torch.tensor(agent_state, dtype=torch.float32).unsqueeze(0).to(self.device)
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
                try:
                    next_state, reward, done = env.step(action)
                except StopIteration:
                    print(f"[Episode {episode}] StopIteration occurred inside step()")
                    done = True
                    break

                current_position, execution_strength = self.split_position_strength(action)

                if type(next_state) == tuple:
                    ts_state, agent_state = next_state
                    if ts_state.ndim == 2:
                        ts_state = torch.tensor(ts_state, dtype=torch.float32).unsqueeze(0).to(self.device)
                    if agent_state.ndim == 1:
                        agent_state = torch.tensor(agent_state, dtype=torch.float32).unsqueeze(0).to(self.device)
                    # print(ts_state.shape)
                    # print(agent_state.shape)
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
                self.pnls.append((float(env.account.unrealized_pnl), float(env.account.net_realized_pnl)))
            
            # save model dicts 
            if max_ep_reward <= ep_reward:
                self.reward_king_model = agent.model.state_dict()
                max_ep_reward = ep_reward 

            if episode % self.save_interval == 0:
                self.models_per_steps.append(agent.model.state_dict())

            self.latest_model = agent.model.state_dict()
            
            # update 
            episode_rewards.append(ep_reward)
            realized_pnl_ratios.append(env.account.realized_pnl_ratio)
            
            # 학습 추적 데이터 기록
            loss = None
            if len(self.memory) >= agent.batch_size:
                advantage = agent.cal_advantage(self.memory)
                loss = agent.train(self.memory, advantage)
                self.memory = []  # 학습 후에만 초기화

            action_prop = (ep_n_positions / sum(ep_n_positions) * 100).round(0)

            # update 
            self.train_rewards_history.append(ep_reward)
            self.train_losses_history.append(loss)

            if (loss != None) & ((episode+1) % self.print_log_interval == 0):
                self.log(f"[{self.dataset_flag}|Train] Ep {episode+1:03d} | info: {env.info} | Maintained for: {env.maintained_steps} | PnL Ratio: {env.account.realized_pnl_ratio * 100:.2f} % | Reward: {ep_reward:4.0f} | Loss: {loss:6.3f} | Pos(short/hold/long): {int(action_prop[-1])}% / {int(action_prop[0])}% / {int(action_prop[1])}% | Strength: {ep_execution_strength / max(ep_len,1):.2f} |")
            
            if (episode+1) % self.print_env_log_interval == 0:
                print(env)
                self.log(env.__str__())

            if env.info == 'bankrupt':
                n_bankruptcy += 1
                
                # 시각화 
                if self.save_visual_log:
                    _, ax = plt.subplots(figsize=(12,6))
                    plot_both_pnl_ticks(ax, list(range(len(self.pnls))), self.pnls)
                    plt.tight_layout()

                    path = self.v_path + '/' + f'T{self.dataset_flag}I{n_bankruptcy}'

                    plt.savefig(path)
                    self.log(f"✅ 시각화 저장 완료: {path}")

            if done:
                self.durations.append(env.maintained_steps)
                self.total_pnl.append(env.account.realized_pnl)

                state = env.reset()
                done = False

                if (episode+1) != self.n_iteration:
                    self.pnls = []

        self.n_bankruptcys.append(n_bankruptcy)
        
        # 인터벌별 학습 데이터 저장
        self.train_rewards_history.extend(interval_rewards)
        self.train_losses_history.extend(interval_losses)

        message = ''
        message += f"\n== [Train 결과 요약: Interval {self.dataset_flag}] ==============================\n"
        message += f"  - 총 에피소드 수         : {episode}\n"
        message += f"  - 최대 보상             : {max_ep_reward:.2f}\n"
        message += f"  - 최종 평균 보상         : {np.mean(episode_rewards):.2f}\n"
        message += f"  - 파산 횟수             : {n_bankruptcy}\n"
        message += f"  - 파산 전 평균 유지 스텝 수: {np.mean(self.durations)}\n"
        message += f"  - 최종 평균 수익률        : {np.mean(realized_pnl_ratios).item() * 100:.2f} %\n"
        message += f"  - 총 실현 수익           : {sum(self.total_pnl)}\n"
        message += "============================================================\n"

        print(message)
        self.log(message)

        self.time_is(start_time, f'Train:I{self.dataset_flag}')
    
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
