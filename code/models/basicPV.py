import torch.nn as nn
import torch.nn.functional as F

class PolicyValueNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        # actor params 
        self.actor_fc1 = nn.Linear(input_size, hidden_size)
        self.actor_fc2 = nn.Linear(hidden_size, output_size)

        # critic params 
        self.critic_fc1 = nn.Linear(input_size, hidden_size)
        self.critic_fc2 = nn.Linear(hidden_size, hidden_size)
        self.critic_fc3 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        actor_x = F.tanh(self.actor_fc1(x))
        policy = F.softmax(self.actor_fc2(actor_x), dim=-1)

        critic_x = F.tanh(self.critic_fc1(x))
        critic_x = F.tanh(self.critic_fc2(critic_x))
        value = self.critic_fc3(critic_x)

        return policy, value