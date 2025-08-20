import torch
import torch.nn as nn
import torch.nn.functional as F

from models.DLinear import *


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

class BasicFusion(nn.Module):
    def __init__(self,
                 input_dim,
                 agent_input_dim,
                 kernel_size,
                 seq_len,
                 pred_len_list,
                 action_size=3,
                 device='cpu',
                 agent_hidden_dim=32,
                 agent_out_dim=32,
                 fusion_hidden_dim=64,
                 num_heads=4,
                 metadata=True,
                 d_ff=64,
                 dropout=0.1):
        super().__init__()
        self.timeseries_block = DLinearModel(num_heads=num_heads,
                                            kernel_size=kernel_size,
                                            seq_len=seq_len,
                                            pred_len_list=pred_len_list,
                                            channel_size=input_dim,
                                            device=device,
                                            decompose_trainable=False,
                                            metadata=metadata)
        
        self.agent_block = AgentModel(agent_input_dim, agent_hidden_dim, agent_out_dim, dropout)

        # Fusion MLP layers
        self.fusion_fc1 = nn.Linear(self.timeseries_block.output_len + agent_out_dim, fusion_hidden_dim)
        self.fusion_relu = nn.ReLU()
        self.fusion_dropout = nn.Dropout(dropout)
        self.fusion_fc2 = nn.Linear(fusion_hidden_dim, fusion_hidden_dim)

    def forward(self, x):
        ts_state, agent_state = x
        
        ts_out = self.timeseries_block(ts_state) # (B, 1, pred_len)
        agent_out = self.agent_block(agent_state)      # (B, agent_out_dim)

        fused = torch.cat([ts_out, agent_out], dim=1)  # (B, pred_len + agent_out)

        x = self.fusion_fc1(fused)
        x = self.fusion_relu(x)
        x = self.fusion_dropout(x)
        x = self.fusion_fc2(x)
        return x

class RegimeFusion(nn.Module):
    def __init__(self,
                 input_dim,
                 agent_input_dim,
                 regime_embed_dim,
                 num_regimes,
                 kernel_size,
                 seq_len,
                 pred_len_list,
                 action_size=3,
                 device='cpu',
                 agent_hidden_dim=32,
                 agent_out_dim=32,
                 fusion_hidden_dim=64,
                 num_heads=4,
                 metadata=True,
                 d_ff=64,
                 dropout=0.1):

        super().__init__()
        self.timeseries_block = DLinearModel(num_heads=num_heads,
                                            kernel_size=kernel_size,
                                            seq_len=seq_len,
                                            pred_len_list=pred_len_list,
                                            channel_size=input_dim,
                                            device=device,
                                            decompose_trainable=False,
                                            metadata=metadata)
        
        self.agent_block = AgentModel(agent_input_dim, agent_hidden_dim, agent_out_dim, dropout)
        self.regime_embedding = nn.Embedding(num_embeddings=num_regimes, embedding_dim=regime_embed_dim)

        # Fusion MLP layers
        self.fusion_fc1 = nn.Linear(self.timeseries_block.output_len + agent_out_dim + regime_embed_dim, fusion_hidden_dim)
        self.fusion_relu = nn.ReLU()
        self.fusion_dropout = nn.Dropout(dropout)
        self.fusion_fc2 = nn.Linear(fusion_hidden_dim, fusion_hidden_dim)

    def forward(self, x):
        """
        market_regime: Tensor of shape (B,) with values in {0: sideways, 1: bull, 2: bear}
        """
        ts_state, agent_state = x
        agent_state, market_regime = agent_state[:,:-1], agent_state[:,-1]

        # mapping: -1 → 2, 0 → 0, 1 → 1
        regime_index = (market_regime == -1).long() * 2 + (market_regime == 1).long() * 1

        ts_out = self.timeseries_block(ts_state)              # (B, output_len)
        agent_out = self.agent_block(agent_state)             # (B, agent_out_dim)
        regime_embed = self.regime_embedding(regime_index)    # (B, regime_embed_dim)

        fused = torch.cat([ts_out, agent_out, regime_embed], dim=1)
        
        x = self.fusion_fc1(fused)
        x = self.fusion_relu(x)
        x = self.fusion_dropout(x)
        x = self.fusion_fc2(x)
        
        return x

class Actor(nn.Module):
    def __init__(self, fusion_hidden_dim, action_size):
        super().__init__()
        # actor params 
        self.actor_fc1 = nn.Linear(fusion_hidden_dim, fusion_hidden_dim)
        self.actor_fc2 = nn.Linear(fusion_hidden_dim, action_size)

        # critic params 
        self.critic_fc1 = nn.Linear(fusion_hidden_dim, fusion_hidden_dim)
        self.critic_fc2 = nn.Linear(fusion_hidden_dim, 1)

    def forward(self, x):
        actor_x = F.tanh(self.actor_fc1(x))
        logits = self.actor_fc2(actor_x)
        return logits

class Critic(nn.Module):
    def __init__(self, fusion_hidden_dim):
        super().__init__()
        # critic params 
        self.critic_fc1 = nn.Linear(fusion_hidden_dim, fusion_hidden_dim)
        self.critic_fc2 = nn.Linear(fusion_hidden_dim, 1)

    def forward(self, x):
        critic_x = F.tanh(self.critic_fc1(x))
        value = self.critic_fc2(critic_x)
        return value 


class MultiStatePV(nn.Module):
    def __init__(self, 
                 input_dim,              # 입력 feature 수 (D)
                 agent_input_dim,        # agent 상태 feature 수
                 kernel_size,
                 seq_len,
                 pred_len_list,
                 action_size, 
                 device,                 # positional encoding에 필요
                 agent_hidden_dim=32, 
                 agent_out_dim=32,
                 fusion_hidden_dim=64,
                 num_heads=4,            # Multi-head attention 헤드 수
                 metadata=True,
                 d_ff=64,                # FFN hidden size
                 dropout=0.1):             # dropout 비율        
        super().__init__()

        self.shared = BasicFusion(input_dim, agent_input_dim,
                                    kernel_size, seq_len, pred_len_list,
                                    action_size, device, 
                                    agent_hidden_dim, agent_out_dim, fusion_hidden_dim,
                                    num_heads, metadata, d_ff, dropout)

        self.actor = Actor(fusion_hidden_dim, action_size)
        self.critic = Critic(fusion_hidden_dim)

    def forward(self, x):
        
        x = self.shared(x)

        logits = self.actor(x)
        value = self.critic(x)
        
        return logits, value

class RegimeAwareMultiStatePV(nn.Module):
    def __init__(self,
                 input_dim,
                 agent_input_dim,
                 kernel_size,
                 seq_len,
                 pred_len_list,
                 regime_embed_dim=4,
                 num_regimes=3,
                 action_size=3,
                 device='cpu',
                 agent_hidden_dim=32,
                 agent_out_dim=32,
                 fusion_hidden_dim=64,
                 num_heads=4,
                 metadata=True,
                 d_ff=64,
                 dropout=0.1):

        super().__init__()

        self.shared = RegimeFusion(input_dim, agent_input_dim, regime_embed_dim, num_regimes,
                                    kernel_size, seq_len, pred_len_list,
                                    action_size, device,
                                    agent_hidden_dim, agent_out_dim, fusion_hidden_dim,
                                    num_heads, metadata, d_ff, dropout)

        self.actor = Actor(fusion_hidden_dim, action_size)
        self.critic = Critic(fusion_hidden_dim)

    def forward(self, x):
        
        x = self.shared(x)

        logits = self.actor(x)
        value = self.critic(x)
        
        return logits, value