#!/usr/bin/env python3 

import numpy as np
import argparse
from copy import deepcopy
import torch

from env_mock import MockEnvironment
from evaluator import Evaluator
from ddpg import DDPG
from util import *

import cv2

WARM_UP_STEPS = 1500

class EnvironmentProxy(object):
    def __init__(self, env, s_shape, act_scale):
        self.env = env
        self.s_shape = s_shape
        self.act_scale = act_scale

    def _process_obs(self, obs):
        obs = cv2.resize(obs, self.s_shape) / 255.0
        return np.transpose(obs, [2, 0, 1])

    def reset(self):
        obs = self.env.reset()
        obs = self._process_obs(obs)
        return obs

    def step(self, act):
        act = np.array([5.0, act])
        for i in range(1):
            obs, reward, done = self.env.step(act)
            obs = self._process_obs(obs)
            if abs(reward) < 1.0:
                reward = 0.0
            if reward <= -1.0:
                done = True
            if done:
                break
        return obs, reward, done, None

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()

def train(num_iterations, gent, env, validate_steps, output, max_episode_length=None, debug=False):

    agent.is_training = True
    step = episode = episode_steps = 0
    episode_reward = 0.
    observation = None

    while step < num_iterations:
        # reset if it is the start of episode
        if observation is None:
            observation = deepcopy(env.reset())
            agent.reset(observation)

        # agent pick action ...
        if step <= WARM_UP_STEPS:
            action = agent.random_action()
        else:
            action = agent.select_action(observation)
        
        # env response with next_observation, reward, terminate_info
        observation2, reward, done, info = env.step(action)
        observation2 = deepcopy(observation2)
        if max_episode_length and episode_steps >= max_episode_length - 1:
            done = True

        # agent observe and update policy
        agent.observe(reward, observation2, done)
        if step > WARM_UP_STEPS :
            agent.update_policy()

        # update 
        step += 1
        episode_steps += 1
        episode_reward += reward
        observation = deepcopy(observation2)
        
        if debug: prGreen('steps:{}, reward:{}, action:{}, done:{}'.format(step, reward, action, done))
        
        if done: 
            # end of episode
            if debug: prGreen('#{}: episode_reward:{} steps:{}'.format(episode,episode_reward,step))

            agent.memory.append(
                observation,
                agent.select_action(observation),
                0., False
            )

            # reset
            observation = None
            episode_steps = 0
            episode_reward = 0.
            episode += 1

if __name__ == "__main__":
    from unity3d_env import Unity3DEnvironment

    nb_states = (3, 64, 64)
    nb_actions = 1

    try:
        agent = DDPG(nb_states, nb_actions)
        env = Unity3DEnvironment()
        env = EnvironmentProxy(env, (64, 64), 10.0)
        train(50000000, agent, env, 
            1000, "log", max_episode_length=100, debug=True)    
    except KeyboardInterrupt:
        env.close()
