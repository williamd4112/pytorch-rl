#!/usr/bin/env python3 

import numpy as np
import argparse
from copy import deepcopy
import torch
import gym

from env_mock import MockEnvironment
from evaluator import Evaluator
from ddpg import DDPG
from util import *

WARM_UP_STEPS = 10

gym.undo_logger_setup()

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
        if max_episode_length and episode_steps >= max_episode_length -1:
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
    nb_states = (1, 64, 64)
    nb_actions = 2
    env = MockEnvironment(nb_states)

    agent = DDPG(nb_states, nb_actions)

    train(1000, agent, env, 
        1000, "log", max_episode_length=100, debug=True)
