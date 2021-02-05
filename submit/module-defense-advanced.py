
import argparse
import gym
import numpy as np
from itertools import count
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'savedAction'))

BATCH_SIZE = 128     
GAMMA = 0.999        
LEARNING_RATE = 0.5
NB_EPISODE = 50_000
NB_ITERATION = 5000
eps = np.finfo(np.float32).eps.item()

class Memory(object):

    def __init__(self, capacity: int):
        self.capacity: int = capacity
        self.memory: list = []
        self.position: int = 0 
    
    def clear(self):
        self.memory: list = []
        self.position: int = 0

    def push(self, *args: list):
        """push a transition"""
        if len(self.memory) < self.capacity:               
            self.memory.append(None)                        
        self.memory[self.position] = Transition(*args)      
        self.position = (self.position + 1) % self.capacity 

    def batch(self, batch_size):
        return self.memory
    
    def __len__(self):
        return len(self.memory)


class Neural(nn.Module):
    def __init__(self, input_size, n_actions):
        super(Neural, self).__init__()
        self.affine1 = nn.Linear(input_size, 128)
        # actor's layer
        self.action_head = nn.Linear(128, n_actions)
        # critic's layer
        self.value_head = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.affine1(x))
        # actor: choses action to take from state s_t 
        # by returning probability of each action
        action_prob = F.softmax(self.action_head(x), dim=-1)
        # critic: evaluates being in the state s_t
        state_values = self.value_head(x)
        # return values for both actor and critic as a tuple of 2 values:
        # 1. a list with the probability of each action over the action space
        # 2. the value from state s_t 
        return action_prob, state_values

class Agent:
    def __init__(self, env, neural, optimizer):
        self.memory = Memory(2000)
        self.env = env
        self.neural = neural
        self.optimizer = optimizer
        self.init()

    def init(self):
        pass
    
    def save(self, path: str):
        pass

    def load(self, path: str):
        pass

    def take_action(self, state) -> [int]:
        state = torch.from_numpy(state).float()
        probs, state_value = self.neural(state)
        # create a categorical distribution over the list of probabilities of actions
        m = Categorical(probs)
        # and sample an action using the distribution
        action = m.sample()
        # the action to take (left or right)
        return action.item(), SavedAction(m.log_prob(action), state_value)          

    def calc_qvals(self, rewards):
        R = 0
        returns = []
        for r in rewards[::-1]:
            R = r + GAMMA * R
            returns.insert(0, R)
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + eps)
        return returns

    def learn(self):
        transitions = self.memory.batch(BATCH_SIZE)
        # On passe d'un tableau de transition a une transition de tableau
        # On met * pour passer l'arg à la fonction zip puis * pour unzip sous forme de list
        batch_rewards = []
        batch: Transition = Transition(*zip(*transitions))
        
        batch_rewards.extend(self.calc_qvals(batch.reward))

        batch_savedActions = batch.savedAction
        batch_rewards = batch_rewards


        policy_losses = [] # list to save actor (policy) loss
        value_losses = [] # list to save critic (value) loss

        for (log_prob, value), R in zip(batch_savedActions, batch_rewards):
            advantage = R - value.item()

            # calculate actor (policy) loss 
            policy_losses.append(-log_prob * advantage)

            # calculate critic (value) loss using L1 smooth loss
            value_losses.append(F.smooth_l1_loss(value, torch.tensor([R])))

        # reset gradients
        self.optimizer.zero_grad()

        # sum up all the values of policy_losses and value_losses
        loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()

        # perform backprop
        loss.backward()
        self.optimizer.step()
        self.memory.clear()

def main():
    env = gym.make("Assault-ram-v0")
    print(env.observation_space)
    print(env.action_space)
    neural = Neural(env.observation_space.shape[0], env.action_space.n)
    optimizer = optim.Adam(neural.parameters(), lr=LEARNING_RATE)
    agent: Agent = Agent(env, neural, optimizer)
    episode_rewards = []
    running_reward = 10
    # run inifinitely many episodes
    for episode in range(NB_EPISODE):
        new_state = env.reset()
        done = False
        episode_rewards = []
        i = 0
        for i in range(1, NB_ITERATION):
            env.render()
            state = new_state
            action, savedAction = agent.take_action(state)
            new_state, reward, done, _ = env.step(action)
            if done == True or i == NB_ITERATION - 1:
                agent.memory.push(state, action, None, reward, savedAction)
                episode_rewards.append(reward)
                agent.learn()
                break
            agent.memory.push(state, action, new_state, reward, savedAction)
            episode_rewards.append(reward)
        print(f"episode: {episode}, iteration {i}, mean {np.mean(episode_rewards)}")

if __name__ == '__main__':
    main()