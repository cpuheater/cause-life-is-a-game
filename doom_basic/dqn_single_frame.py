import numpy as np
from collections import deque
import gym
from gym import spaces
import cv2

cv2.ocl.setUseOpenCL(False)

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import itertools
import argparse
from distutils.util import strtobool
import collections
import numpy as np
import gym
from gym.wrappers import TimeLimit, Monitor
from gym.spaces import Discrete, Box, MultiBinary, MultiDiscrete, Space
from vizdoom import DoomGame, Mode, ScreenFormat, ScreenResolution
import time
import os
import random

from utils import ReplayBuffer, stack_frames

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DQN agent')
    # Common arguments
    parser.add_argument('--exp-name', type=str, default=os.path.basename(__file__).rstrip(".py"),
                        help='the name of this experiment')
    parser.add_argument('--learning-rate', type=float, default=0.00025,
                        help='the learning rate of the optimizer')
    parser.add_argument('--seed', type=int, default=2,
                        help='seed of the experiment')
    parser.add_argument('--total-timesteps', type=int, default=100000,
                        help='total timesteps of the experiments')
    parser.add_argument('--torch-deterministic', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True,
                        help='if toggled, `torch.backends.cudnn.deterministic=False`')
    parser.add_argument('--cuda', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True,
                        help='if toggled, cuda will not be enabled by default')
    parser.add_argument('--wandb-project-name', type=str, default="cleanRL",
                        help="the wandb's project name")
    parser.add_argument('--wandb-entity', type=str, default=None,
                        help="the entity (team) of wandb's project")

    # Algorithm specific arguments
    parser.add_argument('--buffer-size', type=int, default=10000,
                        help='the replay memory buffer size')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='the discount factor gamma')
    parser.add_argument('--target-network-frequency', type=int, default=1000,
                        help="the timesteps it takes to update the target network")
    parser.add_argument('--max-grad-norm', type=float, default=0.5,
                        help='the maximum norm for the gradient clipping')
    parser.add_argument('--batch-size', type=int, default=64,
                        help="the batch size of sample from the reply memory")
    parser.add_argument('--start-e', type=float, default=1.,
                        help="the starting epsilon for exploration")
    parser.add_argument('--end-e', type=float, default=0.02,
                        help="the ending epsilon for exploration")
    parser.add_argument('--exploration-fraction', type=float, default=0.4,
                        help="the fraction of `total-timesteps` it takes from start-e to go end-e")
    parser.add_argument('--learning-starts', type=int, default=800,
                        help="timestep to start learning")
    parser.add_argument('--train-frequency', type=int, default=4,
                        help="the frequency of training")
    args = parser.parse_args()
    #if not args.seed:
    args.seed = int(time.time())

experiment_name = f"{args.exp_name}__{args.seed}__{int(time.time())}"
writer = SummaryWriter(f"runs/{experiment_name}")
writer.add_text('hyperparameters', "|param|value|\n|-|-|\n%s" % (
    '\n'.join([f"|{key}|{value}|" for key, value in vars(args).items()])))

device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')
print(device)

np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = args.torch_deterministic

class Scale(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        return x * self.scale

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class QNetwork(nn.Module):
    def __init__(self, actions, frames=3):
        super(QNetwork, self).__init__()
        self.network = nn.Sequential(
            Scale(1 / 255),
            layer_init(nn.Conv2d(frames, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(2560, 512)),
            nn.ReLU(),
            nn.Linear(512, len(actions))
        )

    def forward(self, x):
        x = torch.Tensor(x).to(device)
        return self.network(x)


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)

def initialize_vizdoom(config):
    game = DoomGame()
    game.load_config(config)
    #game.set_window_visible(False)
    game.set_mode(Mode.PLAYER)
    game.set_screen_format(ScreenFormat.CRCGCB)
    game.set_screen_resolution(ScreenResolution.RES_160X120)
    game.init()
    return game

def preprocess(img, resolution):
    img = img.transpose((1, 2, 0))
    return torch.from_numpy(cv2.resize(img, resolution).astype(np.float32).transpose((2,0,1)))

frame_repeat = 4
resolution = (64, 112)
frames = 3
game = initialize_vizdoom("./scenarios/basic.cfg")
#stacked_frames = deque([np.zeros((64, 112), dtype=np.int) for i in range(frames)], maxlen=4)
n = game.get_available_buttons_size()
actions = [list(a) for a in itertools.product([0, 1], repeat=n)]

rb = ReplayBuffer(args.buffer_size)
q_network = QNetwork(actions, frames).to(device)
target_network = QNetwork(actions, frames).to(device)
target_network.load_state_dict(q_network.state_dict())
optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
loss_fn = nn.MSELoss()
print(device.__repr__())
print(q_network)

game.new_episode()
episode_reward = 0
obs = game.get_state().screen_buffer
obs = preprocess(obs, resolution)
for global_step in range(args.total_timesteps):
    epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)
    if random.random() < epsilon:
        action = torch.tensor(random.randint(0, len(actions) - 1)).long().item()
    else:
        logits = q_network.forward(obs.reshape((1,) + obs.shape))
        action = torch.argmax(logits, dim=1).tolist()[0]
    reward = game.make_action(actions[action], frame_repeat)
    reward *= 0.01
    done = game.is_episode_finished()

    next_obs = preprocess(np.zeros((3, resolution[0], resolution[1])), resolution) if done else preprocess(game.get_state().screen_buffer, resolution)

    if done:
        writer.add_scalar("charts/episode_reward", game.get_total_reward(), global_step)
        writer.add_scalar("charts/epsilon", epsilon, global_step)

    rb.put((obs, action, reward, next_obs, done))
    if global_step > args.learning_starts and global_step % args.train_frequency == 0:
        s_obs, s_actions, s_rewards, s_next_obses, s_dones = rb.sample(args.batch_size)
        with torch.no_grad():
            target_max = torch.max(target_network.forward(s_next_obses), dim=1)[0]
            td_target = torch.Tensor(s_rewards).to(device) + args.gamma * target_max * (
                    1 - torch.Tensor(s_dones).to(device))
        old_val = q_network.forward(s_obs).gather(1, torch.LongTensor(s_actions).view(-1, 1).to(device)).squeeze()
        loss = loss_fn(td_target, old_val)

        if global_step % 100 == 0:
            writer.add_scalar("losses/td_loss", loss, global_step)

        optimizer.zero_grad()
        loss.backward()
        #nn.utils.clip_grad_norm_(list(q_network.parameters()), args.max_grad_norm)
        optimizer.step()

        if global_step % args.target_network_frequency == 0:
            target_network.load_state_dict(q_network.state_dict())

    if done:
        game.new_episode()
        obs, episode_reward = game.get_state().screen_buffer, 0
        obs = preprocess(obs, resolution)
    else:
        obs = next_obs

writer.close()
