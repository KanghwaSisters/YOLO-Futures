import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
# refer to Section 3.5 in the paper

    def __init__(self, device, max_len=512, d_model=16):
        super().__init__()
        self.pos_enc = torch.zeros(max_len,d_model,requires_grad=False, device=device)
        pos = torch.arange(0, max_len, 1, requires_grad=False, device=device).reshape(-1,1)
        w_vector = 10000**(-2*(torch.arange(0, (d_model // 2), 1, device=device))/d_model)

        self.pos_enc[:,0::2] = torch.cos(pos * w_vector)
        self.pos_enc[:,1::2] = torch.sin(pos * w_vector)


    def forward(self, x):
        """
        x.shape = [batch_size, seq_len, data_dim]
        """
        return x + self.pos_enc[:x.shape[1], :].unsqueeze(0)
    
class ScaledDotProductAttention(nn.Module):
    def __init__(self,d_model=16):
        super().__init__()
        self.d_model = d_model

    def forward(self, q, k, v, mask=None):
        """
        q, k, v = transformed query, key, value
        q.shape, k.shape, v.shpae = [batch_size, num_head, seq_len, d_ff=d_model/num_head]
        mask = masking matrix, if the index has value False, kill the value; else, leave the value
        """
        k_T = k.transpose(-1,-2)

        # 1. matmul Q @ K_T
        scores = (q @ k_T) / math.sqrt(q.shape[-1])

        # ( Optional ) masking
        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(~mask, float('-inf'))

        # 2. softmax
        attention_weight = F.softmax(scores, dim=-1)

        # 3. matmul attention_weight @ V
        attention_value = attention_weight @ v

        return attention_value
    
class MultiHeadAttention(nn.Module):
    def __init__(self,d_model=16,num_head=4):
        super().__init__()
        assert d_model % num_head == 0, "check if d_model is divisible by num_head"

        # params
        self.d_model = d_model
        self.num_head = num_head
        self.d_ff = d_model//num_head

        # q, k, v's weight
        self.q_weight = nn.Linear(d_model, d_model)
        self.k_weight = nn.Linear(d_model, d_model)
        self.v_weight = nn.Linear(d_model, d_model)

        # output weight for concat
        self.output_weight =  nn.Linear(d_model, d_model)

        # set attention block
        self.attention = ScaledDotProductAttention(d_model=d_model)

    def forward(self, q, k, v, mask=None):
        # compute multi-head attention value
        """
        q, k, v = pre-transformed query, key, value
        q.shape, k.shape, v.shpae = [batch_size, seq_len, d_model]
        mask = masking matrix, if the index has value False, kill the value; else, leave the value
        """
        batch_size, seq_len, d_model = q.shape

        # make them learnable
        q, k, v = self.q_weight(q), self.k_weight(k), self.v_weight(v)

        # reshape [batch_size, seq_len, d_model] to [batch_size, num_head, seq_len, d_ff]
        def reshape(x):
            return x.view(batch_size, seq_len, self.num_head, self.d_ff).transpose(1,2)

        q, k, v = reshape(q), reshape(k), reshape(v)

        # calculate attention value
        attention_value = self.attention(q,k,v,mask=mask)

        # concat heads --> result :  [batch_size, seq_len, d_model]
        concated_value = attention_value.transpose(1,2).reshape(batch_size, seq_len, d_model)

        output = self.output_weight(concated_value)

        return output
    
class PositionwiseFeedForwardNetwork(nn.Module):
    def __init__(self,d_model=16,d_ff=32):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
class LayerNormalization(nn.Module):
    def __init__(self,d_model=16,eps=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self,x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)

        normed = (x - mean)/torch.sqrt(var + self.eps) # 정규화
        normed = self.gamma * normed + self.beta # 파라미터 추가

        return normed
    
class EncoderLayer(nn.Module):
    # Pre-Norm 구조로 기존의 Post-Norm 구조와 다르다. 
    def __init__(self,d_model=16,num_head=4,d_ff=32,drop_prob=.1):
        super().__init__()
        self.norm1 = LayerNormalization(d_model)
        self.attention = MultiHeadAttention(d_model, num_head)

        self.norm2 = LayerNormalization(d_model)
        self.ffn = PositionwiseFeedForwardNetwork(d_model, d_ff)

        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self,enc):
        # multi head attention
        _x = enc
        x = self.norm1(_x)
        x = self.attention(q=x, k=x, v=x)

        # add and norm
        x = self.dropout(x)
        x = x + _x

        # feed forward
        _x = x
        x = self.norm2(_x)
        x = self.ffn(x)

        # add and norm
        x = self.dropout(x)
        x = x + _x

        return x
    
class Encoder(nn.Module):
    def __init__(self,device,input_dim=3,num_layer=3,max_len=512,d_model=16,num_head=4,d_ff=32,drop_prob=.1):
        super().__init__()
        self.positional_emb = PositionalEncoding(device=device,
                                                 max_len=max_len,
                                                 d_model=d_model)

        self.layers = nn.ModuleList([EncoderLayer(d_model=d_model,
                                                  num_head=num_head,
                                                  d_ff=d_ff,
                                                  drop_prob=drop_prob)
                                                  for _ in range(num_layer)])

        self.input_fc = nn.Linear(input_dim, d_model)


    def forward(self,x):
        # transform dimension : embedding이 없어서 필요한 부분
        x = self.input_fc(x)

        x = self.positional_emb(x)

        for layer in self.layers:
            hidden = layer(x)

        return hidden
    
class CNNTokenizer(nn.Module):
    def __init__(self, input_dim, embed_dim, kernel_size=4, stride=1):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=input_dim, out_channels=embed_dim,
                              kernel_size=kernel_size, stride=stride)

    def forward(self, x):
        # x shape: (B, T, D) → (B, D, T)
        x = x.permute(0, 2, 1)
        x = self.conv(x)  # (B, embed_dim, N_tokens)
        x = x.permute(0, 2, 1)  # (B, N_tokens, embed_dim)
        return x
    

class CTTS(nn.Module):
    def __init__(self, 
                 input_dim,              # 입력 feature 수 (D)
                 embed_dim,              # CNN + Transformer 임베딩 차원 (d_model)
                 kernel_size,            # CNN 커널 사이즈
                 stride,                 # CNN stride
                 device,                 # positional encoding에 필요
                 num_layers=3,           # Transformer 층 수
                 num_heads=4,            # Multi-head attention 헤드 수
                 d_ff=64,                # FFN hidden size
                 dropout=0.1):           # dropout 비율
        super().__init__()

        self.tokenizer = CNNTokenizer(input_dim=input_dim, 
                                      embed_dim=embed_dim, 
                                      kernel_size=kernel_size, 
                                      stride=stride)

        self.encoder = Encoder(device=device,
                               input_dim=embed_dim,   # CNN output = Transformer input
                               num_layer=num_layers,
                               max_len=128,           # CNN 이후 토큰 개수의 upper bound
                               d_model=embed_dim,
                               num_head=num_heads,
                               d_ff=d_ff,
                               drop_prob=dropout)

        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # x: (B, T=30, D)
        tokens = self.tokenizer(x)              # (B, N_tokens, embed_dim)
        encoded = self.encoder(tokens)          # (B, N_tokens, embed_dim)

        # 대표 토큰을 사용 (e.g., 평균)
        pooled = encoded.mean(dim=1)            # (B, embed_dim)

        out = self.head(pooled)                 # (B, embed_dim)
        return out
    
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
                 embed_dim=64,
                 kernel_size=3,
                 stride=1,
                 action_size=3,
                 device='cpu',
                 agent_hidden_dim=32,
                 agent_out_dim=32,
                 fusion_hidden_dim=64,
                 num_layers=3,
                 num_heads=4,
                 d_ff=64,
                 dropout=0.1):
        super().__init__()
        self.timeseries_block = CTTS(input_dim, embed_dim, 
                                     kernel_size, stride, device, 
                                     num_layers, num_heads, 
                                     d_ff, dropout)
        
        self.agent_block = AgentModel(agent_input_dim, agent_hidden_dim, agent_out_dim, dropout)

        # Fusion MLP layers
        self.fusion_fc1 = nn.Linear(embed_dim + agent_out_dim, fusion_hidden_dim)
        self.fusion_relu = nn.ReLU()
        self.fusion_dropout = nn.Dropout(dropout)
        self.fusion_fc2 = nn.Linear(fusion_hidden_dim, fusion_hidden_dim)

    def forward(self, x):
        ts_state, agent_state = x

        ts_out = self.timeseries_block(ts_state)       # (B, embed_dim)
        agent_out = self.agent_block(agent_state)      # (B, agent_out_dim)

        fused = torch.cat([ts_out, agent_out], dim=1)  # (B, embed + agent_out)

        x = self.fusion_fc1(fused)
        x = self.fusion_relu(x)
        x = self.fusion_dropout(x)
        x = self.fusion_fc2(x)
        return x

class RegimeFusion(nn.Module):
    def __init__(self,
                 input_dim,
                 agent_input_dim,
                 regime_embed_dim=4,
                 num_regimes=3,
                 embed_dim=64,
                 kernel_size=3,
                 stride=1,
                 action_size=3,
                 device='cpu',
                 agent_hidden_dim=32,
                 agent_out_dim=32,
                 fusion_hidden_dim=64,
                 num_layers=3,
                 num_heads=4,
                 d_ff=64,
                 dropout=0.1):

        super().__init__()
        self.timeseries_block = CTTS(input_dim, embed_dim, 
                                     kernel_size, stride, device, 
                                     num_layers, num_heads, 
                                     d_ff, dropout)
        
        self.agent_block = AgentModel(agent_input_dim, agent_hidden_dim, agent_out_dim, dropout)
        self.regime_embedding = nn.Embedding(num_embeddings=num_regimes, embedding_dim=regime_embed_dim)

        # Fusion MLP layers
        self.fusion_fc1 = nn.Linear(embed_dim + agent_out_dim + regime_embed_dim, fusion_hidden_dim)
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

        ts_out = self.timeseries_block(ts_state)              # (B, embed_dim)
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
                 embed_dim,              # CNN + Transformer 임베딩 차원 (d_model)
                 kernel_size,            # CNN 커널 사이즈
                 stride,                 # CNN stride
                 action_size, 
                 device,                 # positional encoding에 필요
                 agent_hidden_dim=32, 
                 agent_out_dim=32,
                 fusion_hidden_dim=64,
                 num_layers=3,           # Transformer 층 수
                 num_heads=4,            # Multi-head attention 헤드 수
                 d_ff=64,                # FFN hidden size
                 dropout=0.1):             # dropout 비율        
        super().__init__()

        self.shared = BasicFusion(input_dim, agent_input_dim, embed_dim, kernel_size,           
                                    stride, action_size, device, 
                                    agent_hidden_dim, agent_out_dim, fusion_hidden_dim,
                                    num_layers, num_heads, d_ff, dropout)

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
                 regime_embed_dim=4,
                 num_regimes=3,
                 embed_dim=64,
                 kernel_size=3,
                 stride=1,
                 action_size=3,
                 device='cpu',
                 agent_hidden_dim=32,
                 agent_out_dim=32,
                 fusion_hidden_dim=64,
                 num_layers=3,
                 num_heads=4,
                 d_ff=64,
                 dropout=0.1):

        super().__init__()

        self.shared = RegimeFusion(input_dim, agent_input_dim, regime_embed_dim, num_regimes, embed_dim,            
                                    kernel_size, stride, action_size, device, 
                                    agent_hidden_dim, agent_out_dim, fusion_hidden_dim,
                                    num_layers, num_heads, d_ff, dropout)

        self.actor = Actor(fusion_hidden_dim, action_size)
        self.critic = Critic(fusion_hidden_dim)

    def forward(self, x):
        
        x = self.shared(x)

        logits = self.actor(x)
        value = self.critic(x)
        
        return logits, value