import sys

import torch
import gymnasium
from time import sleep
import numpy as np
import torch.nn as nn
import torch
from torch.distributions.categorical import  Categorical
from dataclasses import dataclass
import tyro
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader
import vizdoom as vzd
import cv2
from tqdm.auto import tqdm
from ppo import Agent, create_actions
from vizdoom import GameVariable, Button

@dataclass
class Args:
    env_id: str = "battle"
    """the id of the environment"""
    num_actions: int = 12
    """num actions"""
    num_episodes: int = 5
    """"""
    model_file: str = "agent_2000000.pt"
    """"""
    skip_frame: int = 2
    """"""
    render: bool = True
    """"""

if __name__ == '__main__':
    args = tyro.cli(Args)
    game = vzd.DoomGame()
    game.load_config(f"./scenarios/{args.env_id}.cfg")
    game.set_window_visible(False)
    game.set_mode(vzd.Mode.ASYNC_PLAYER)

    if args.render:
        game.set_window_visible(True)
        game.set_mode(vzd.Mode.ASYNC_PLAYER)
    else:
        game.set_mode(vzd.Mode.PLAYER)

    game.init()
    model = Agent(args.num_actions, 3)
    model = torch.load(args.model_file)
    model.eval()
    total_rewards = []
    possible_actions = create_actions(np.array([
            Button.TURN_LEFT, Button.TURN_RIGHT, Button.MOVE_FORWARD, Button.MOVE_BACKWARD, Button.ATTACK]))
    for episode in tqdm(range(args.num_episodes)):
        game.new_episode()
        obs = game.get_state().screen_buffer
        while True:
            obs = cv2.resize(obs, (112, 64), interpolation = cv2.INTER_AREA)
            obs = obs.transpose(2,0,1)
            obs = torch.from_numpy(obs).float().unsqueeze(0)
            _, action, _, _ = model.get_action(obs)
            action = action.detach().numpy()[0]
            game.make_action(possible_actions[action], args.skip_frame)
            if game.is_episode_finished():
                total_rewards.append(game.get_total_reward())
                print(f"Total reward: {game.get_total_reward()}, running mean: {np.mean(total_rewards)}")
                break
            else:
                obs = game.get_state().screen_buffer
    print(f"Mean reward: {np.mean(total_rewards)} using {args.model_file}")
