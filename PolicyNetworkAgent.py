import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from collections import deque


class NeuralNet(nn.Module):
    def __init__(self, input_dim, output_dim, device, hidden_dim, softmax_output):
        super(NeuralNet, self).__init__()
        self.softmax_output = softmax_output
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
        if self.softmax_output:
            x = torch.softmax(self.fc3(x), dim=-1)
        else:
            x = self.fc3(x)
        return x


class REINFORCEAgent:
    def __init__(self, n_actions, n_states, alpha, gamma, hidden_dim, normalize):
        self.gamma = gamma
        self.normalize = normalize

        # Set the device to GPU if available, otherwise CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pi = NeuralNet(n_states, n_actions, self.device, hidden_dim, softmax_output=True)
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


class ActorCriticAgent:
    def __init__(self, n_actions, n_states, alpha, gamma, hidden_dim, estim_depth, update_episodes):
        self.gamma = gamma
        self.estim_depth = estim_depth
        self.update_episodes = update_episodes

        # Set the device to GPU if available, otherwise CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pi = NeuralNet(n_states, n_actions, self.device, hidden_dim, softmax_output=True)
        self.optimizer_pi = optim.Adam(self.pi.parameters(), lr=alpha)  # Adam optimizer
        self.V = NeuralNet(n_states, 1, self.device, hidden_dim, softmax_output=False)
        self.optimizer_V = optim.Adam(self.V.parameters(), lr=alpha)  # Adam optimizer

        self.__reset_update_buffer()

    def __reset_update_buffer(self):
        self.update_count = 0
        self.update_states = torch.empty(0, dtype=torch.float, device=self.device)
        self.update_actions = torch.empty(0, dtype=torch.int, device=self.device)
        self.Q_hat = torch.empty(0, dtype=torch.float, device=self.device)

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
        if self.estim_depth > len(trace_rewards) - 1:
            V_target_states = torch.tensor(trace_states[self.estim_depth:], dtype=torch.float, device=self.device)
            with torch.no_grad():
                V_target_pred = self.V.forward(V_target_states)

            n_step_rewards = [trace_rewards[i:i+self.estim_depth] for i in range(len(trace_rewards) - self.estim_depth)]
            n_step_rewards = torch.tensor(n_step_rewards, dtype=torch.float, device=self.device)

            n_step_returns = V_target_pred
            for step_rewards in reversed(n_step_rewards.T):
                n_step_returns *= self.gamma
                n_step_returns += step_rewards

            self.Q_hat = torch.cat((self.Q_hat, n_step_returns))

        def mc_discounted_returns(rewards):
            returns = deque()
            R_t = 0
            for r in reversed(rewards):
                R_t = r + self.gamma * R_t
                returns.appendleft(R_t)
            return returns

        terminal_rewards = trace_rewards[-self.estim_depth:]
        terminal_returns = torch.tensor(mc_discounted_returns(terminal_rewards), dtype=torch.float, device=self.device)

        self.Q_hat = torch.cat((self.Q_hat, terminal_returns))

        self.update_states = torch.cat((self.update_states, trace_states))
        self.update_actions = torch.cat((self.update_actions, trace_actions))

        self.update_count += 1
        if self.update_count < self.update_episodes:
            return

        V_current = self.V.forward(self.update_states)
        V_loss = F.mse_loss(self.Q_hat, V_current, reduction='sum')

        self.optimizer_V.zero_grad()
        V_loss.backward()
        self.optimizer_V.step()

        pi_action_probs = self.pi.forward(self.update_states)
        pi_m = Categorical(pi_action_probs)
        pi_log_probs = pi_m.log_prob(trace_actions)
        pi_loss = (-1 * pi_log_probs * self.Q_hat).sum()

        self.optimizer_pi.zero_grad()
        pi_loss.backward()
        self.optimizer_pi.step()

        self.__reset_update_buffer()
