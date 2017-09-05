
import numpy as np

import torch
import torch.nn as nn
from torch.optim import Adam

from model import (Actor, Critic)
from memory import SequentialMemory
from random_process import OrnsteinUhlenbeckProcess
from util import *

# from ipdb import set_trace as debug

ACTOR_LR = 1e-4
CRITIC_LR = 1e-3
MEMORY_SIZE = int(1e6)
HISTORY_LEN = 1
BATCH_SIZE = 16
OU_THETA = 0.5
OU_SIGMA = 0.3
OU_MU = 0.0
TAU = 1e-2
GAMMA = 0.99
DEPSILON = 100000.0

criterion = nn.MSELoss()

class DDPG(object):
    def __init__(self, nb_states, nb_actions):
        self.nb_states = nb_states
        self.nb_actions= nb_actions
        
        # Create Actor and Critic Network
        self.actor = Actor(self.nb_states, self.nb_actions)
        self.actor_target = Actor(self.nb_states, self.nb_actions)
        self.actor_optim  = Adam(self.actor.parameters(), lr=ACTOR_LR)

        self.critic = Critic(self.nb_states, self.nb_actions)
        self.critic_target = Critic(self.nb_states, self.nb_actions)
        self.critic_optim  = Adam(self.critic.parameters(), lr=CRITIC_LR)

        hard_update(self.actor_target, self.actor) # Make sure target is with the same weight
        hard_update(self.critic_target, self.critic)
        
        #Create replay buffer
        self.memory = SequentialMemory(limit=MEMORY_SIZE, window_length=HISTORY_LEN)
        self.random_process = OrnsteinUhlenbeckProcess(size=nb_actions, theta=OU_THETA, mu=OU_MU, sigma=OU_SIGMA)

        # Hyper-parameters
        self.batch_size = BATCH_SIZE
        self.tau = TAU
        self.discount = GAMMA
        self.depsilon = 1.0 / DEPSILON

        self.epsilon = 1.0
        self.s_t = None # Most recent state
        self.a_t = None # Most recent action
        self.is_training = True

        if USE_CUDA: self.cuda()

    def update_policy(self):
        # Sample batch
        state_batch, action_batch, reward_batch, \
        next_state_batch, terminal_batch = self.memory.sample_and_split(self.batch_size)

        # Prepare for the target q batch
        next_q_values = self.critic_target([
            to_tensor(next_state_batch, volatile=True),
            self.actor_target(to_tensor(next_state_batch, volatile=True)),
        ])[:, 0]
        next_q_values.volatile=False

        target_q_batch = to_tensor(reward_batch) + \
            self.discount*to_tensor(terminal_batch.astype(np.float))*next_q_values

        # Critic update
        self.critic.zero_grad()

        q_batch = self.critic([to_tensor(state_batch), to_tensor(action_batch) ])
        
        value_loss = criterion(q_batch, target_q_batch)
        value_loss.backward()
        
        torch.nn.utils.clip_grad_norm(self.critic.parameters(), 10.0)
        for p in self.critic.parameters():
            p.data.add_(-CRITIC_LR, p.grad.data)
        self.critic_optim.step()

        # Actor update  
        self.actor.zero_grad()

        policy_loss = -self.critic([
            to_tensor(state_batch),
            self.actor(to_tensor(state_batch))
        ])

        policy_loss = policy_loss.mean()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm(self.actor.parameters(), 10.0)
        for p in self.actor.parameters():
            p.data.add_(-ACTOR_LR, p.grad.data)
        self.actor_optim.step()

        # Target update
        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)

    def eval(self):
        self.actor.eval()
        self.actor_target.eval()
        self.critic.eval()
        self.critic_target.eval()

    def cuda(self):
        self.actor.cuda()
        self.actor_target.cuda()
        self.critic.cuda()
        self.critic_target.cuda()

    def observe(self, r_t, s_t1, done):
        if self.is_training:
            self.memory.append(self.s_t, self.a_t, r_t, done)
            self.s_t = s_t1

    def random_action(self):
        action = np.random.uniform(-1.,1.,self.nb_actions)
        self.a_t = action
        return action

    def select_action(self, s_t, decay_epsilon=True):
        action = to_numpy(
            self.actor(to_tensor(np.array([s_t])))
        )[0]
        ou = self.random_process.sample()

        prGreen('eps:{}, act:{}, random:{}'.format(self.epsilon, action, ou))
        action += self.is_training * max(self.epsilon, 0) * ou
        action = np.clip(action, -1., 1.)

        if decay_epsilon:
            self.epsilon -= self.depsilon
        
        self.a_t = action
        return action

    def reset(self, obs):
        self.s_t = obs
        self.random_process.reset_states()

    def load_weights(self, output):
        if output is None: return

        self.actor.load_state_dict(
            torch.load('{}/actor.pkl'.format(output))
        )

        self.critic.load_state_dict(
            torch.load('{}/critic.pkl'.format(output))
        )


    def save_model(self,output):
        torch.save(
            self.actor.state_dict(),
            '{}/actor.pkl'.format(output)
        )
        torch.save(
            self.critic.state_dict(),
            '{}/critic.pkl'.format(output)
        )

    def seed(self,s):
        torch.manual_seed(s)
        if USE_CUDA:
            torch.cuda.manual_seed(s)
