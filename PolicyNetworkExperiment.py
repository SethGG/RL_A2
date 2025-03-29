import numpy as np
import gymnasium as gym
from Helper import LearningCurvePlot, smooth
from PolicyNetworkAgent import REINFORCEAgent
import os
import torch.multiprocessing as mp


def evaluation(agent: REINFORCEAgent):
    # Evaluate the agent's performance in the environment
    env = gym.make('CartPole-v1')
    s, info = env.reset()
    done = False
    trunc = False
    episode_return = 0
    while not done and not trunc:
        a = agent.select_action_greedy(s)
        s_next, r, done, trunc, info = env.step(a)
        episode_return += r
        s = s_next
    return episode_return


def run_single_repetition(task):
    config_id, rep_id, n_envsteps, eval_interval, params = task
    alpha = params["alpha"]
    gamma = params["gamma"]
    hidden_dim = params["hidden_dim"]

    env = gym.make('CartPole-v1')
    n_actions = env.action_space.n
    n_states = env.observation_space.shape[0]
    eval_returns = np.zeros(int(n_envsteps / eval_interval))

    agent = REINFORCEAgent(n_actions, n_states, alpha, gamma, hidden_dim)

    envstep = 0
    eval_num = 0
    while envstep < n_envsteps:
        s, info = env.reset()
        done = False
        trunc = False

        trace_states = []
        trace_actions = []
        trace_rewards = []
        while not done and not trunc:
            trace_states.append(s)
            a = agent.select_action_sample(s)
            trace_actions.append(a)
            s, r, done, trunc, info = env.step(a)
            trace_rewards.append(r)

            envstep += 1
            if envstep % eval_interval == 0:
                eval_return = evaluation(agent)
                eval_returns[eval_num] = eval_return
                eval_num += 1

                print(f"Running config: {config_id+1:2}, Repetition {rep_id+1:2}, Environment steps: {envstep:6}, "
                      f"Eval return: {eval_return:3}")

            if envstep == n_envsteps:
                break

        agent.update(trace_states, trace_actions, trace_rewards)

    # de echte gradient descent moet in een update function in the agent
    # dingen om te testen: met en zonder normalization, alleen de volledige return gebruiken


if __name__ == '__main__':
    run_single_repetition((0, 0, 50000, 1000, {'alpha': 0, 'gamma': 1, 'hidden_dim': 128}))
