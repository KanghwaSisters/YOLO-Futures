import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque  # 고정 크기 버퍼 구현에 사용

# Q-Network 정의: 상태(state)를 입력받아 각 행동(action)에 대한 Q값 출력하는 신경망
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        # 2개의 fully connected layer로 구성
        self.layers = nn.Sequential(
            nn.Linear(state_dim, 128),  # 입력층(state_dim) → 은닉층(128 유닛)
            nn.ReLU(),                  # 활성화 함수 ReLU
            nn.Linear(128, action_dim)  # 은닉층 → 출력층(action_dim), 각 행동별 Q값 출력
        )

    def forward(self, x):
        # 네트워크의 순전파(포워드) 함수: 입력 x에 대해 Q값 출력
        return self.layers(x)


# DQN 에이전트 클래스
class DQNAgent:
    def __init__(self,
                 state_dim,         # 상태 공간 차원 (예: CartPole은 4)
                 action_dim,        # 행동 공간 크기 (예: CartPole은 2)
                 gamma=0.99,        # 할인율, 미래 보상에 대한 현재 가치 반영 비율
                 lr=1e-3,           # 학습률
                 batch_size=32,     # 학습 미니배치 크기
                 buffer_size=10000, # 경험 리플레이 버퍼 최대 크기
                 epsilon=1.0,       # 탐험 확률 초기값 (epsilon-greedy 정책)
                 epsilon_min=0.01,  # epsilon의 최소값 (더 이상 줄이지 않음)
                 epsilon_decay=0.999, # epsilon 감소 비율 (에피소드 혹은 스텝마다 감소)
                 target_update_freq=10,  # 타겟 네트워크 업데이트 주기(몇 번 학습 후 업데이트)
                 device='cpu'):     # 연산 디바이스 설정(cpu 또는 cuda)

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.lr = lr
        self.batch_size = batch_size

        # 고정 크기 큐 형태의 경험 리플레이 버퍼
        self.buffer = deque(maxlen=buffer_size)

        # epsilon-greedy 정책 관련 변수
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.device = device

        # Q-Network 및 타겟 네트워크 초기화 (동일 구조, 다른 가중치)
        self.q_net = QNetwork(state_dim, action_dim).to(device)
        self.target_q_net = QNetwork(state_dim, action_dim).to(device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())  # 초기 가중치 동기화

        # 옵티마이저: Adam, 손실 함수: MSELoss
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.lr)
        self.loss_fn = nn.MSELoss()

        # 학습 중 손실 기록용 리스트
        self.loss_history = []

        # 타겟 네트워크 업데이트 주기 및 카운터 초기화
        self.target_update_freq = target_update_freq
        self.update_count = 0

    def get_action(self, state, epsilon=None):
        """
        현재 상태에 대해 행동 선택 (epsilon-greedy)
        - epsilon 확률로 랜덤 행동 (탐험)
        - 1 - epsilon 확률로 Q-Network가 예측한 최적 행동 (이용)
        """
        if epsilon is None:
            epsilon = self.epsilon

        if np.random.rand() < epsilon:
            return np.random.randint(self.action_dim)  # 랜덤 행동
        else:
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)  # 배치 차원 추가
            q_values = self.q_net(state)  # Q값 예측
            return torch.argmax(q_values, dim=1).item()  # 최대 Q값 행동 반환

    def update(self):
        """
        Q-Network 학습 수행
        - 버퍼에서 미니배치 샘플링
        - 벨만 최적 방정식에 따른 타겟 Q값 계산
        - MSE Loss 계산 및 역전파
        - 타겟 네트워크를 일정 주기마다 업데이트
        - epsilon 감소
        """
        if len(self.buffer) < self.batch_size:
            return  # 버퍼 부족 시 학습 중지

        batch = random.sample(self.buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # numpy 배열 → 텐서 변환 및 장치 이동
        states = torch.FloatTensor(np.vstack(states)).to(self.device)
        next_states = torch.FloatTensor(np.vstack(next_states)).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # 현재 상태-행동 쌍의 Q값
        q_values = self.q_net(states).gather(1, actions)
        # 다음 상태에서 최대 Q값 (타겟 네트워크 사용, 그래디언트 차단)
        next_q_values = self.target_q_net(next_states).max(dim=1, keepdim=True)[0].detach()
        # 타겟 Q값 계산 (종료 상태면 보상만, 아니면 보상+감가된 미래 최대 Q값)
        target = rewards + (1 - dones) * self.gamma * next_q_values

        # 손실 계산 및 기록
        loss = self.loss_fn(q_values, target)
        self.loss_history.append(loss.item())

        # 역전파 및 가중치 업데이트
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 학습 횟수 증가 및 타겟 네트워크 업데이트 조건 검사
        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.update_target()

        # epsilon 점진적 감소
        self.decay_epsilon()

    def remember(self, state, action, reward, next_state, done):
        """
        경험 리플레이 버퍼에 transition 저장
        - state, next_state는 numpy float32 배열로 저장하여 일관성 유지
        """
        self.buffer.append((
            np.array(state, dtype=np.float32),
            action,
            reward,
            np.array(next_state, dtype=np.float32),
            done
        ))

    def update_target(self):
        """
        타겟 네트워크를 현재 Q-Network 가중치로 동기화
        """
        self.target_q_net.load_state_dict(self.q_net.state_dict())

    def decay_epsilon(self):
        """
        epsilon 값을 점진적으로 감소시키되, epsilon_min 이하로는 내려가지 않음
        """
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
