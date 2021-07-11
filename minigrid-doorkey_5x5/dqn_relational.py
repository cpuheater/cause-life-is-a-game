# https://github.com/facebookresearch/torchbeast/blob/master/torchbeast/core/environment.py

import numpy as np
from collections import deque
import gym
from gym import spaces
import cv2
cv2.ocl.setUseOpenCL(False)
from einops import rearrange

class ImageToPyTorch(gym.ObservationWrapper):
    """
    Image shape to channels x weight x height
    """

    def __init__(self, env):
        super(ImageToPyTorch, self).__init__(env)
        old_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(old_shape[-1], old_shape[0], old_shape[1]),
            dtype=np.uint8,
        )

    def observation(self, observation):
        return np.transpose(observation, axes=(2, 0, 1))


def wrap_pytorch(env):
    return ImageToPyTorch(env)

def wrap_env(env):
    env = InfoWrapper(env)
    env = WarpFrame(env)
    env = wrap_pytorch(env)
    return env

class InfoWrapper(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self._rewards = []

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self._rewards = []
        return obs["image"]

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._rewards.append(reward)
        ## Retrieve the RGB frame of the agent's vision
        vis_obs = obs["image"]

        ## Render the environment in realtime
        #if self._realtime_mode:
        #    self._env.render(tile_size=96)
        #    time.sleep(0.5)

        # Wrap up episode information once completed (i.e. done)
        if done:
            info = {"reward": sum(self._rewards),
                    "length": len(self._rewards)}

        return vis_obs, reward, done, info

class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env, width=84, height=84):
        super().__init__(env)
        self.observation_space = env.observation_space.spaces['image']

    def observation(self, obs):
        return obs


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import argparse
from distutils.util import strtobool
import collections
import numpy as np
import gym
from gym.wrappers import TimeLimit, Monitor
from gym.spaces import Discrete, Box, MultiBinary, MultiDiscrete, Space
import time
import random
import os
from gym_minigrid.wrappers import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DQN agent')
    # Common arguments
    parser.add_argument('--exp-name', type=str, default=os.path.basename(__file__).rstrip(".py"),
                        help='the name of this experiment')
    parser.add_argument('--gym-id', type=str, default="MiniGrid-DoorKey-5x5-v0",
                        help='the id of the gym environment')
    parser.add_argument('--learning-rate', type=float, default=0.0005,
                        help='the learning rate of the optimizer')
    parser.add_argument('--seed', type=int, default=2,
                        help='seed of the experiment')
    parser.add_argument('--total-timesteps', type=int, default=10000000,
                        help='total timesteps of the experiments')
    parser.add_argument('--torch-deterministic', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
                        help='if toggled, `torch.backends.cudnn.deterministic=False`')
    parser.add_argument('--cuda', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
                        help='if toggled, cuda will not be enabled by default')
    parser.add_argument('--prod-mode', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
                        help='run the script in production mode and use wandb to log outputs')
    parser.add_argument('--capture-video', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
                        help='weather to capture videos of the agent performances (check out `videos` folder)')
    parser.add_argument('--wandb-project-name', type=str, default="cleanRL",
                        help="the wandb's project name")
    parser.add_argument('--wandb-entity', type=str, default=None,
                        help="the entity (team) of wandb's project")

    # Algorithm specific arguments
    parser.add_argument('--buffer-size', type=int, default=9000,
                        help='the replay memory buffer size')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='the discount factor gamma')
    parser.add_argument('--target-network-frequency', type=int, default=100,
                        help="the timesteps it takes to update the target network")
    parser.add_argument('--max-grad-norm', type=float, default=1.0,
                        help='the maximum norm for the gradient clipping')
    parser.add_argument('--batch-size', type=int, default=32,
                        help="the batch size of sample from the reply memory")
    parser.add_argument('--start-e', type=float, default=1.,
                        help="the starting epsilon for exploration")
    parser.add_argument('--end-e', type=float, default=0.02,
                        help="the ending epsilon for exploration")
    parser.add_argument('--exploration-fraction', type=float, default=0.10,
                        help="the fraction of `total-timesteps` it takes from start-e to go end-e")
    parser.add_argument('--learning-starts', type=int, default=80000,
                        help="timestep to start learning")
    parser.add_argument('--train-frequency', type=int, default=4,
                        help="the frequency of training")
    args = parser.parse_args()
    if not args.seed:
        args.seed = int(time.time())

# TRY NOT TO MODIFY: setup the environment
experiment_name = f"{args.gym_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
writer = SummaryWriter(f"runs/{experiment_name}")
writer.add_text('hyperparameters', "|param|value|\n|-|-|\n%s" % (
    '\n'.join([f"|{key}|{value}|" for key, value in vars(args).items()])))
if args.prod_mode:
    import wandb
    wandb.init(project=args.wandb_project_name, entity=args.wandb_entity, sync_tensorboard=True, config=vars(args), name=experiment_name, monitor_gym=True, save_code=True)
    writer = SummaryWriter(f"/tmp/{experiment_name}")

# TRY NOT TO MODIFY: seeding
device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')
#device = "cpu"
env = gym.make(args.gym_id)
env = wrap_env(env)
env = gym.wrappers.RecordEpisodeStatistics(env) # records episode reward in `info['episode']['r']`
if args.capture_video:
    env = Monitor(env, f'videos/{experiment_name}')

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = args.torch_deterministic
env.seed(args.seed)
env.action_space.seed(args.seed)
env.observation_space.seed(args.seed)
# respect the default timelimit
assert isinstance(env.action_space, Discrete), "only discrete action space is supported"

# modified from https://github.com/seungeunrho/minimalRL/blob/master/dqn.py#
class ReplayBuffer():
    def __init__(self, buffer_limit):
        self.buffer = collections.deque(maxlen=buffer_limit)

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append(a)
            r_lst.append(r)
            s_prime_lst.append(s_prime)
            done_mask_lst.append(done_mask)

        return np.array(s_lst), np.array(a_lst), \
               np.array(r_lst), np.array(s_prime_lst), \
               np.array(done_mask_lst)

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

def prepare_state(x):
    maxv = x.flatten().max()
    x = x / maxv
    return x

# ALGO LOGIC: initialize agent here:
class Scale(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        return x * self.scale

class MultiHeadRelationalModule(torch.nn.Module):
    def __init__(self):
        super(MultiHeadRelationalModule, self).__init__()
        self.conv1_ch = 16
        self.conv2_ch = 20
        self.node_size = 64
        self.lin_hid = 100
        self.out_dim = 5
        self.ch_in = 3
        self.sp_coord_dim = 2
        self.N = int(7 ** 2)
        self.n_heads = 3

        self.conv1 = layer_init(nn.Conv2d(self.ch_in, self.conv1_ch, kernel_size=(1, 1), padding=0))  # A
        self.conv2 = layer_init(nn.Conv2d(self.conv1_ch, self.conv2_ch, kernel_size=(1, 1), padding=0))
        self.proj_shape = (self.conv2_ch + self.sp_coord_dim, self.n_heads * self.node_size)
        self.k_proj = layer_init(nn.Linear(*self.proj_shape))
        self.q_proj = layer_init(nn.Linear(*self.proj_shape))
        self.v_proj = layer_init(nn.Linear(*self.proj_shape))

        self.k_lin = layer_init(nn.Linear(self.node_size, self.N))  # B
        self.q_lin = layer_init(nn.Linear(self.node_size, self.N))
        self.a_lin = layer_init(nn.Linear(self.N, self.N))

        self.node_shape = (self.n_heads, self.N, self.node_size)
        self.k_norm = nn.LayerNorm(self.node_shape, elementwise_affine=True)
        self.q_norm = nn.LayerNorm(self.node_shape, elementwise_affine=True)
        self.v_norm = nn.LayerNorm(self.node_shape, elementwise_affine=True)

        self.linear1 = layer_init(nn.Linear(self.n_heads * self.node_size, self.node_size))
        self.norm1 = nn.LayerNorm([self.N, self.node_size], elementwise_affine=False)
        self.linear2 = nn.Linear(self.node_size, self.out_dim)

    def forward(self, x):
        N, Cin, H, W = x.shape
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = torch.relu(x)
        with torch.no_grad():
            self.conv_map = x.clone()  # C
        _, _, cH, cW = x.shape
        xcoords = torch.arange(cW).repeat(cH, 1).float().to(device) / cW
        ycoords = torch.arange(cH).repeat(cW, 1).transpose(1, 0).float().to(device) / cH
        spatial_coords = torch.stack([xcoords, ycoords], dim=0)
        spatial_coords = spatial_coords.unsqueeze(dim=0)
        spatial_coords = spatial_coords.repeat(N, 1, 1, 1)
        x = torch.cat([x, spatial_coords], dim=1)
        x = x.permute(0, 2, 3, 1)
        x = x.flatten(1, 2)

        K = rearrange(self.k_proj(x), "b n (head d) -> b head n d", head=self.n_heads)
        K = self.k_norm(K)

        Q = rearrange(self.q_proj(x), "b n (head d) -> b head n d", head=self.n_heads)
        Q = self.q_norm(Q)

        V = rearrange(self.v_proj(x), "b n (head d) -> b head n d", head=self.n_heads)
        V = self.v_norm(V)
        A = torch.nn.functional.elu(self.q_lin(Q) + self.k_lin(K))  # D
        A = self.a_lin(A)
        A = torch.nn.functional.softmax(A, dim=3)
        with torch.no_grad():
            self.att_map = A.clone()  # E
        E = torch.einsum('bhfc,bhcd->bhfd', A, V)  # F
        E = rearrange(E, 'b head n d -> b n (head d)')
        E = self.linear1(E)
        E = torch.relu(E)
        E = self.norm1(E)
        E = E.max(dim=1)[0]
        y = self.linear2(E)
        y = torch.nn.functional.elu(y)
        return y


class QNetwork(nn.Module):
    def __init__(self, env, frames=3):
        super(QNetwork, self).__init__()
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(frames, 16, kernel_size=(1, 1), padding=0)),
            nn.ReLU(),
            layer_init(nn.Conv2d(16, 20, kernel_size=(1, 1), padding=0)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(980, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, env.action_space.n)
        )

    def forward(self, x, device):
        x = torch.Tensor(x).to(device)
        return self.network(x)

def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope =  (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)

action_map = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    4: 5,
}

rb = ReplayBuffer(args.buffer_size)
q_network = MultiHeadRelationalModule().to(device)
target_network = MultiHeadRelationalModule().to(device)
target_network.load_state_dict(q_network.state_dict())
optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
loss_fn = nn.MSELoss()
print(device.__repr__())
print(q_network)

# TRY NOT TO MODIFY: start the game
obs = prepare_state(env.reset())
episode_reward = 0
for global_step in range(args.total_timesteps):
    # ALGO LOGIC: put action logic here
    epsilon = 0.5#linear_schedule(args.start_e, args.end_e, args.exploration_fraction*args.total_timesteps, global_step)
    #obs = np.array(obs)
    if random.random() < epsilon:
        action = int(torch.randint(0, 5, size=(1,)).squeeze())
    else:
        logits = q_network.forward(torch.Tensor(obs.reshape((1,)+obs.shape)).to(device))
        action = torch.argmax(logits, dim=1).tolist()[0]

    # TRY NOT TO MODIFY: execute the game and log data.
    next_obs, reward, done, info = env.step(action_map[action])
    next_obs = prepare_state(next_obs)
    reward = -0.01 if reward == 0 else reward
    episode_reward += reward

    # TRY NOT TO MODIFY: record rewards for plotting purposes
    if 'episode' in info.keys():
        print(f"global_step={global_step}, episode_reward={info['episode']['r']}")
        writer.add_scalar("charts/episode_reward", info['episode']['r'], global_step)
        writer.add_scalar("charts/epsilon", epsilon, global_step)

    # ALGO LOGIC: training.
    rb.put((obs, action, reward, next_obs, done))
    if global_step > args.learning_starts and global_step % args.train_frequency == 0:
        s_obs, s_actions, s_rewards, s_next_obses, s_dones = rb.sample(args.batch_size)
        with torch.no_grad():
            target_max = torch.max(target_network.forward(torch.Tensor(s_next_obses).to(device)), dim=1)[0]
            td_target = torch.Tensor(s_rewards).to(device) + args.gamma * target_max * (1 - torch.Tensor(s_dones).to(device))
        old_val = q_network.forward(torch.Tensor(s_obs).to(device)).gather(1, torch.LongTensor(s_actions).view(-1,1).to(device)).squeeze()
        loss = loss_fn(td_target, old_val)

        if global_step % 100 == 0:
            writer.add_scalar("losses/td_loss", loss, global_step)

        # optimize the midel
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(q_network.parameters(), args.max_grad_norm)
        optimizer.step()

        # update the target network
        if global_step % args.target_network_frequency == 0:
            target_network.load_state_dict(q_network.state_dict())

    # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
    obs = next_obs
    if done:
        # important to note that because `EpisodicLifeEnv` wrapper is applied,
        # the real episode reward is actually the sum of episode reward of 5 lives
        # which we record through `info['episode']['r']` provided by gym.wrappers.RecordEpisodeStatistics
        obs, episode_reward = prepare_state(env.reset()), 0

env.close()
writer.close()
