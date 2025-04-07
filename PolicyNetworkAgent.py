import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from collections import deque


class NeuralNet(nn.Module):
    def __init__(self, input_dim, output_dim, device, hidden_dim):
        super(NeuralNet, self).__init__()
        # Define the layers of the neural network
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

        # Move the model to the specified device (CPU or GPU)
        self.to(device)

    def forward(self, x):
        # Define the forward pass
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.softmax(self.fc3(x), dim=-1)
        return x


class REINFORCEAgent:
    def __init__(self, n_actions, n_states, alpha, gamma, hidden_dim, normalize):
        self.gamma = gamma
        self.normalize = normalize

        # Set the device to GPU if available, otherwise CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pi = NeuralNet(n_states, n_actions, self.device, hidden_dim)
        self.optimizer = optim.Adam(self.pi.parameters(), lr=alpha)  # Adam optimizer

    def select_action_sample(self, state):
        state = torch.tensor(state, dtype=torch.float, device=self.device)
        with torch.no_grad():
            action_probs = self.pi.forward(state)
        m = Categorical(action_probs)
        action = m.sample().item()
        return action

    def select_action_greedy(self, state):
        state = torch.tensor(state, dtype=torch.float, device=self.device)
        with torch.no_grad():
            action_probs = self.pi.forward(state)
        action = torch.argmax(action_probs).item()
        return action

    def update(self, trace_states, trace_actions, trace_rewards):
        trace_returns = deque()
        R_t = 0
        for r in reversed(trace_rewards):
            R_t = r + self.gamma * R_t
            trace_returns.appendleft(R_t)
        trace_returns = torch.tensor(trace_returns, dtype=torch.float, device=self.device)

        # normalize returns
        if self.normalize:
            trace_returns = (trace_returns - trace_returns.mean()) / trace_returns.std()

        trace_states = torch.tensor(trace_states, dtype=torch.float, device=self.device)
        trace_actions = torch.tensor(trace_actions, dtype=torch.int, device=self.device)

        trace_action_probs = self.pi.forward(trace_states)
        trace_m = Categorical(trace_action_probs)
        trace_log_probs = trace_m.log_prob(trace_actions)

        policy_loss = (-1 * trace_log_probs * trace_returns).sum()

        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()
