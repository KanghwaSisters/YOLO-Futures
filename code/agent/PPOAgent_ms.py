import torch
import torch.nn as nn
import random
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

class PPOAgent:
    '''
    PPOAgent(action_space: Any,
             n_actions: int,
             model: nn.Module,
             value_coeff: float,
             entropy_coeff: float,
             clip_eps: float,
             gamma: float,
             lr: float,
             lam: float = 0.98) -> PPOAgent

    ----------
    Proximal Policy Optimization(PPO) 알고리즘을 구현한 에이전트 클래스.

    - policy/value를 동시에 출력하는 네트워크를 사용한다.
    - clipped surrogate objective와 GAE를 통해 안정적인 학습을 수행한다.

    memory 구조:
        list[tuple[
            torch.Tensor,  # state: shape = [1, state_dim]
            torch.Tensor,  # action: shape = [1] or [1, 1]
            torch.Tensor,  # reward: shape = [1]
            torch.Tensor,  # next_state: shape = [1, state_dim]
            torch.Tensor,  # done: shape = [1], 1이면 종료, 0이면 계속
            torch.Tensor   # log_prob: shape = [1]
        ]]

    예시:
        (
            tensor([[0.1, 0.2]]),   # state
            tensor([1]),           # action
            tensor([0.5]),         # reward
            tensor([[0.3, 0.4]]),  # next_state
            tensor([0]),           # done
            tensor([-0.69])        # log_prob
        )
    '''
    def __init__(self, action_space, n_actions, 
                model, value_coeff, entropy_coeff, clip_eps, 
                gamma, lr, batch_size, epoch, device, 
                lam=0.98, lambda_entry=0.1, kappa=2.0, 
                beta=1.0, regulation=3.0):
        '''
        PPOAgent 클래스 초기화 함수.

        모델, PPO 관련 계수들, 옵티마이저를 초기화한다.
        '''
        self.model = model.to(device)
        self.device = device

        # action params 
        self.action_space = action_space
        self.n_actions = n_actions

        # coeffs • epsilon 
        self.value_coeff = value_coeff
        self.entropy_coeff = entropy_coeff
        self.clip_eps = clip_eps

        self.lambda_entry = lambda_entry
        self.kappa = kappa
        self.beta = beta
        self.regulation = regulation 

        # discount params 
        self.gamma = gamma
        self.lam = lam

        # train related 
        self.lr = lr
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.critic_loss_ftn = nn.MSELoss()
        self.epoch = epoch
        self.batch_size = batch_size

    def get_action(self, state, mask=None, stochastic=True):
        '''
        get_action(state: torch.Tensor) -> tuple[int, float]

        ----------
        주어진 상태로부터 행동을 샘플링하고 로그 확률을 반환한다.

        - policy에서 확률 분포를 생성하고 행동을 샘플링한다.
        - 샘플링된 행동의 로그 확률도 함께 반환한다.
        '''
        state = tuple(s.to(self.device) for s in state)
        logits, _ = self.model(state)

        # mask: shape [n_actions] with 1 (valid) or 0 (invalid)
        if mask is not None:
            mask = torch.tensor(mask, dtype=torch.bool).unsqueeze(0).to(self.device)
            logits = logits.masked_fill(mask == 0, float('-inf'))

        if stochastic:
            # entropy bonus 
            action_dist = Categorical(logits=logits)
            _action = action_dist.sample()
            log_prob = action_dist.log_prob(_action)

            action = self.action_space[_action.item()]
            return action, log_prob.item(), logits
        else:
            _action = torch.argmax(logits, dim=-1)
            action = self.action_space[_action.item()]
            return action, None, None


    def clip_loss_ftn(self, advantage, old_prob, current_prob):
        '''
        clip_loss_ftn(advantage: torch.Tensor,
                      old_prob: torch.Tensor,
                      current_prob: torch.Tensor) -> torch.Tensor

        ----------
        PPO의 clipped surrogate loss를 계산한다.

        - 현재 확률 대비 이전 확률의 비율을 계산하고,
          clip 범위 안에서 surrogate loss를 구한다.
        - 안정적인 policy 업데이트를 위함이다.
        '''
        ratio = current_prob / (old_prob + 1e-8)
        clipped_ratio = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps)
        surrogate1 = ratio * advantage
        surrogate2 = clipped_ratio * advantage
        return torch.min(surrogate1, surrogate2).mean()

    def cal_advantage(self, memory, lam=0.95):
        '''
        cal_advantage(memory: list[tuple], lam: float) -> torch.Tensor

        ----------
        Generalized Advantage Estimation(GAE)를 계산한다.

        - reversed list로 delta -> gae를 계산한다. 
        - GAE를 사용하면 bias-variance trade-off를 조절할 수 있다.
        '''
        # set memory
        states, _, rewards, next_states, dones, _, _, _, _, _ = zip(*memory)

        # zip again to separate ts / agent
        ts_states, ag_states = zip(*states)
        n_ts_states, n_ag_states = zip(*next_states)

        # cat across batch dimension
        ts_states = torch.cat(ts_states, dim=0).to(self.device)
        ag_states = torch.cat(ag_states, dim=0).to(self.device)
        n_ts_states = torch.cat(n_ts_states, dim=0).to(self.device)
        n_ag_states = torch.cat(n_ag_states, dim=0).to(self.device)

        states = (ts_states, ag_states)
        next_states = (n_ts_states, n_ag_states)
        rewards = torch.cat(rewards).view(-1)
        dones = torch.cat(dones).view(-1)

        # get values - next_values : GAE 계산을 위함 
        with torch.no_grad():
            _, values = self.model(states)
            _, next_values = self.model(next_states)

        values = values.squeeze().detach()
        next_values = next_values.squeeze().detach()

        # Generalize Advantage Estimate(GAE) calculation
        # reversed list로 delta -> gae를 계산한다. 
        advantage = []
        gae = 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * next_values[t] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * lam * (1 - dones[t]) * gae
            advantage.insert(0, gae)

        return torch.tensor(advantage, dtype=torch.float32).unsqueeze(1)
    
    def _get_max_power2_batch_size(self, len_data, min_n=4, max_n=9):
        """
        len_data 이하에서 사용할 수 있는 최대 2^n 배치 크기를 반환.
        예: len_data=90 → 64 반환 (2^6)
        """
        for n in reversed(range(min_n, max_n + 1)):
            batch_size = 2 ** n
            if batch_size <= len_data:
                return batch_size
        return 2 ** min_n  # 기본값 (len_data가 너무 작을 경우)
    
    
    def sample_memory(self, memory, advantage):
        # actual batch size
        # actual_batch_size = self._get_max_power2_batch_size(len(memory))

        # sampling from raw data 
        indices = random.sample(range(len(memory)), self.batch_size)
        sampled_memory = [memory[i] for i in indices]

        states, actions, rewards, next_states, dones, old_log_probs, masks, entry_masks, entry_scores, log_policy = zip(*sampled_memory)
        advantages = advantage[indices].to(self.device)

        # zip again to separate ts / agent
        ts_states, ag_states = zip(*states)
        n_ts_states, n_ag_states = zip(*next_states)

        # cat across batch dimension
        ts_states = torch.cat(ts_states, dim=0).to(self.device)
        ag_states = torch.cat(ag_states, dim=0).to(self.device)
        n_ts_states = torch.cat(n_ts_states, dim=0).to(self.device)
        n_ag_states = torch.cat(n_ag_states, dim=0).to(self.device)

        states = (ts_states, ag_states)
        next_states = (n_ts_states, n_ag_states)

        actions = torch.cat(actions)
        rewards = torch.cat(rewards).to(self.device)
        dones = torch.cat(dones).to(self.device)
        old_log_probs = torch.cat(old_log_probs).unsqueeze(1).to(self.device)
        masks = torch.cat(masks).to(self.device)
        entry_masks = torch.cat(entry_masks).to(self.device)
        entry_scores = torch.cat(entry_scores).to(self.device)
        log_policies = torch.cat(log_policy).to(self.device)

        # invert action indices
        offset = -self.action_space[0]          # ex : -(-5) = 5
        actions = (actions + offset).to(self.device)  

        return states, actions, rewards, next_states, dones, old_log_probs, advantages, masks, entry_masks, entry_scores, log_policies


    def train(self, memory, advantage):
        '''
        train(memory: list[tuple], advantage: torch.Tensor) -> float

        ----------
        PPO 손실 함수를 계산하고 모델 파라미터를 업데이트한다.

        - 세 가지 손실 항을 포함한다: 
          (1) value loss, (2) clipped surrogate loss, (3) entropy bonus
        - GAE로 계산된 advantage를 기반으로 policy와 value를 모두 학습한다.
        '''
        if len(memory) < self.batch_size:
            return
        
        losses = 0

        for _ in range(self.epoch):
            # set memory
            states, actions, rewards, next_states, dones, old_log_probs, advantages, masks, entry_masks, entry_scores, log_policies = self.sample_memory(memory, advantage)

            # get current values 
            self.model.train()
            current_logits, values = self.model(states)

            if masks is not None:
                # mask: shape [n_actions] with 1 (valid) or 0 (invalid)
                current_logits = current_logits.masked_fill(masks == 0, float('-inf'))

            # entropy bonus 
            action_dist = Categorical(logits=current_logits)                        
            current_log_probs = action_dist.log_prob(actions.squeeze()).unsqueeze(1)
            current_probs = current_log_probs.exp()

            # KL Divergence 
            # [1] 현재 롱 숏 방향 분포 
            position_cap = self.n_actions // 2

            current_policy_logit = current_logits.exp()
            entry_probs = current_policy_logit[entry_masks]

            short_probs = entry_probs[:, :position_cap].sum(dim=1)
            long_probs = entry_probs[:, position_cap:].sum(dim=1)

            total_probs = torch.stack([short_probs, long_probs], dim=1)
            sum_probs = total_probs.sum(dim=1, keepdim=True).clamp_min(1e-8)
            current_entry_policy = total_probs / sum_probs

            # [2] 상태별 트렌드 점수 s_t
            s = entry_scores[entry_masks]
            p_long = torch.sigmoid(self.kappa * s)
            p_target = torch.stack([1-p_long, p_long], dim=1)

            # [3] 극단치 방지를 위해 uniform prior와 섞음 
            p_mix = self.beta * 0.5 + (1-self.beta) * p_target

            # [4] trend가 확실하다면 규제를 약화 
            w = torch.sigmoid(-self.regulation * torch.abs(s)).detach()

            # [5] KL( 현재 정책 | 타깃 혼합 분포 )
            policy_safe = current_entry_policy.clamp_min(1e-8)
            p_mix_safe = p_mix.clamp_min(1e-8)
            kl = (policy_safe * (policy_safe.log() - p_mix_safe.log())).sum(dim=1)
            entry_reg = (w * kl).mean() 

            # 3 elements of loss : value_loss, clip_loss, entropy bonus 
            with torch.no_grad():
                _, next_values = self.model(next_states)
                value_target = rewards + self.gamma * next_values.squeeze() * (1 - dones)

            value_loss = self.critic_loss_ftn(values.squeeze(), value_target.detach())
            clip_loss = self.clip_loss_ftn(advantages.detach(), old_log_probs.exp().detach(), current_probs)
            entropy = action_dist.entropy().mean()

            total_loss = -clip_loss + self.value_coeff * value_loss - self.entropy_coeff * entropy + self.lambda_entry * entry_reg

            losses += total_loss.item()

            # back-propagation 
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

        return losses / self.epoch
        

    def load_model(self, state_dict):
        self.model.load_state_dict(state_dict)

    def set_optimizer(self, new_optimizer):
        self.optimizer = new_optimizer(self.model.parameters(), lr=self.lr)

class DecoupledPPOAgent(PPOAgent):
    def __init__(self, action_space, n_actions, model, value_coeff, entropy_coeff, clip_eps, gamma, lr, batch_size, epoch, device, lam=0.98):
        super().__init__(action_space, n_actions, model, value_coeff, entropy_coeff, clip_eps, gamma, lr, batch_size, epoch, device, lam)

        self.shared_params = list(self.model.shared.parameters())
        self.actor_params = list(self.model.actor.parameters())
        self.critic_params = list(self.model.critic.parameters())

        self.actor_optimizer = torch.optim.Adam(self.shared_params + self.actor_params, lr=self.lr)
        self.critic_optimizer = torch.optim.Adam(self.shared_params + self.critic_params, lr=self.lr)

    def train(self, memory, advantage):
        if len(memory) < self.batch_size:
            return

        total_loss_sum = 0.0

        for _ in range(self.epoch):
            states, actions, rewards, next_states, dones, old_log_probs, advantages, masks, entry_masks, entry_scores, log_policies = self.sample_memory(memory, advantage)

            self.model.train()
            current_logits, values = self.model(states)

            if masks is not None:
                current_logits = current_logits.masked_fill(masks == 0, float('-inf'))

            action_dist = Categorical(logits=current_logits)
            current_log_probs = action_dist.log_prob(actions.squeeze()).unsqueeze(1)
            current_probs = current_log_probs.exp()

            with torch.no_grad():
                _, next_values = self.model(next_states)
                value_target = rewards + self.gamma * next_values.squeeze() * (1 - dones)

            # KL Divergence 
            # [1] 현재 롱 숏 방향 분포 
            if entry_masks.sum() == 0:
                entry_reg = torch.tensor(0.0, device=self.device)
            else:
                position_cap = self.n_actions // 2
                # if sum(entry_masks) == 0:
                current_policy_logit = current_logits.exp()
                entry_probs = current_policy_logit[entry_masks]

                short_probs = entry_probs[:, :position_cap].sum(dim=1, keepdim=True)
                long_probs = entry_probs[:, position_cap:].sum(dim=1, keepdim=True)

                total_probs = torch.cat([short_probs, long_probs], dim=1)

                sum_probs = total_probs.sum(dim=1, keepdim=True).clamp_min(1e-8)
                current_entry_policy = total_probs / sum_probs

                # [2] 상태별 트렌드 점수 s_t
                s = entry_scores[entry_masks]
                p_long = torch.sigmoid(self.kappa * s)
                p_target = torch.stack([1-p_long, p_long], dim=1)

                # [3] 극단치 방지를 위해 uniform prior와 섞음 
                p_mix = self.beta * 0.5 + (1-self.beta) * p_target

                # [4] trend가 확실하다면 규제를 약화 
                w = torch.sigmoid(-self.regulation * torch.abs(s)).detach()

                # [5] KL( 현재 정책 | 타깃 혼합 분포 )
                policy_safe = current_entry_policy.clamp_min(1e-8)
                p_mix_safe = p_mix.clamp_min(1e-8)
                kl = (policy_safe * (policy_safe.log() - p_mix_safe.log())).sum(dim=1)

                entry_reg = (w * kl).mean() 

            # Critic loss (MSE)
            critic_loss = self.value_coeff * self.critic_loss_ftn(values.squeeze(), value_target.detach())

            # critic update 
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            # nn.utils.clip_grad_norm_(self.critic_params, max_grad_norm)
            self.critic_optimizer.step()

            # Actor loss (clipped PPO surrogate)
            clip_loss = self.clip_loss_ftn(advantages.detach(), old_log_probs.exp().detach(), current_probs)
            entropy = action_dist.entropy().mean()
            actor_loss = -clip_loss - self.entropy_coeff * entropy + self.lambda_entry * entry_reg

            # Sum loss for tracking
            total_loss = actor_loss + critic_loss 
            total_loss_sum += total_loss.item()

            # Backprop & update
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            # nn.utils.clip_grad_norm_(self.actor_params, max_grad_norm)
            self.actor_optimizer.step()

        return total_loss_sum / self.epoch