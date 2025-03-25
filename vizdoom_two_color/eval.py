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
from ppo_lstm import Agent, get_actions
import vizdoom as vzd
import cv2
from tqdm.auto import tqdm

@dataclass
class Args:
    env_id: str = "k_item"
    """the id of the environment"""
    num_actions: int = 17
    """num actions"""
    num_episodes: int = 5
    """"""
    model_file: str = "models/agent_10000000.pt"
    """"""
    skip_frame: int = 4
    """"""
    render: bool = True
    """"""
    rnn_hidden_size = 512

if __name__ == '__main__':
    args = tyro.cli(Args)
    game = vzd.DoomGame()
    game.load_config(f"scenarios/{args.env_id}.cfg")
    game.set_window_visible(False)
    game.set_mode(vzd.Mode.ASYNC_PLAYER)

    if args.render:
        game.set_window_visible(True)
        game.set_mode(vzd.Mode.ASYNC_PLAYER)
    else:
        game.set_mode(vzd.Mode.PLAYER)

    game.init()
    actions = get_actions()
    model = Agent(args.num_actions, args.rnn_hidden_size, 3)
    model.load_state_dict(torch.load(args.model_file, weights_only = False))
    model.eval()
    total_rewards = []
    for episode in tqdm(range(args.num_episodes)):
        game.new_episode()
        obs = game.get_state().screen_buffer
        rnn_state = (torch.zeros(1, 1, args.rnn_hidden_size), torch.zeros(1, 1, args.rnn_hidden_size))
        num_step = 0
        while True:
            obs = cv2.resize(obs, (112, 64), interpolation = cv2.INTER_AREA)
            obs = obs.transpose(2,0,1)
            obs = torch.from_numpy(obs).float().unsqueeze(0)
            rnn_state, out = model.get_logits(obs, rnn_state)
            probs = torch.sigmoid(out.squeeze(0))
            max_idx = torch.argmax(probs, 0, keepdim=True)
            game.make_action(actions[max_idx], args.skip_frame)
            num_step += 1    
            if game.is_episode_finished():
                total_rewards.append(game.get_total_reward())
                print(f"Total reward: {game.get_total_reward()}, running mean: {np.mean(total_rewards)}, steps: f{num_step}")
                break
            else:
                obs = game.get_state().screen_buffer
    print(f"Mean reward: {np.mean(total_rewards)} using {args.model_file}")
