# https://github.com/facebookresearch/torchbeast/blob/master/torchbeast/core/environment.py

import numpy as np
from collections import deque
import vizdoom
import gymnasium
from gymnasium import spaces
from gymnasium.spaces import Discrete, Box
import cv2
from vizdoom import DoomGame, Mode, ScreenFormat, ScreenResolution
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import cv2
cv2.ocl.setUseOpenCL(False)
import argparse
from distutils.util import strtobool
import collections
import numpy as np
import time
import random
import os
import itertools


class ViZDoomEnv(gymnasium.Env):
    def __init__(self,
                 game: vizdoom.DoomGame, channels = 1):
        super(ViZDoomEnv, self).__init__()
        self.action_space = spaces.Discrete(game.get_available_buttons_size())
        h, w = game.get_screen_height(), game.get_screen_width()
        IMAGE_WIDTH, IMAGE_HEIGHT = 112, 64
        self.observation_space = spaces.Box(low=0, high=255, shape=(channels, IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.uint8)

        # Assign other variables
        self.game = game
        self.possible_actions = np.eye(self.action_space.n).tolist()  # VizDoom needs a list of buttons states.
        self.frame_skip = args.frame_skip
        self.scale_reward = args.scale_reward
        self.empty_frame = np.zeros(self.observation_space.shape, dtype=np.uint8)
        self.state = self.empty_frame
        self.channels = channels

    def step(self, action: int):
        info = {}
        reward = self.game.make_action(self.possible_actions[action], self.frame_skip)
        done = self.game.is_episode_finished()
        self.state = self._get_frame(done)
        reward = reward * self.scale_reward
        self.total_reward += reward
        self.total_length += 1
        if done:
            info['reward'] = self.total_reward
            info['length'] = self.total_length

        return self.state, reward, done, info

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self.game.set_seed(seed)
        self.game.new_episode()
        self.state = self._get_frame()
        self.total_reward = 0
        self.total_length = 0
        return self.state, {}

    def close(self) -> None:
        self.game.close()

    def render(self, mode='human'):
        pass

    def get_screen(self):
        screen = self.game.get_state().screen_buffer
        channels, h, w = self.observation_space.shape
        screen = cv2.resize(screen, (w, h), cv2.INTER_AREA)
        if screen.ndim == 2:
            screen = np.expand_dims(screen, 0)
        else:
            screen = screen.transpose(2, 0, 1)
        return screen

    def _get_frame(self, done: bool = False) -> np.ndarray:
        return self.get_screen() if not done else self.empty_frame


def create_env() -> ViZDoomEnv:
    game = vizdoom.DoomGame()
    game.load_config(f'scenarios/{args.env_id}.cfg')
    game.set_window_visible(True)
    game.init()
    return ViZDoomEnv(game, channels=args.channels)



class Scale(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        return x * self.scale

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DQN agent')
    # Common arguments
    parser.add_argument('--exp-name', type=str, default=os.path.basename(__file__).rstrip(".py"),
                        help='the name of this experiment')
    parser.add_argument('--env-id', type=str, default="basic",
                        help='the id of the gym environment')
    parser.add_argument('--learning-rate', type=float, default=0.0001,
                        help='the learning rate of the optimizer')
    parser.add_argument('--seed', type=int, default=2,
                        help='seed of the experiment')
    parser.add_argument('--total-timesteps', type=int, default=100000,
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
    parser.add_argument('--scale-reward', type=float, default=0.01,
                        help='scale reward')
    parser.add_argument('--channels', type=int, default=3,
                        help="the number of channels")
    parser.add_argument('--frame-skip', type=int, default=4,
                        help='frame skip')

    # Algorithm specific arguments
    parser.add_argument('--buffer-size', type=int, default=10000,
                        help='the replay memory buffer size')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='the discount factor gamma')
    parser.add_argument('--target-network-frequency', type=int, default=1000,
                        help="the timesteps it takes to update the target network")
    parser.add_argument('--max-grad-norm', type=float, default=0.5,
                        help='the maximum norm for the gradient clipping')
    parser.add_argument('--batch-size', type=int, default=128,
                        help="the batch size of sample from the reply memory")
    parser.add_argument('--start-e', type=float, default=1.0,
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

# TRY NOT TO MODIFY: setup the environment
experiment_name = f"{args.exp_name}__{args.seed}__{int(time.time())}"
writer = SummaryWriter(f"runs/{experiment_name}")
writer.add_text('hyperparameters', "|param|value|\n|-|-|\n%s" % (
    '\n'.join([f"|{key}|{value}|" for key, value in vars(args).items()])))
if args.prod_mode:
    import wandb
    wandb.init(project=args.wandb_project_name, entity=args.wandb_entity, sync_tensorboard=True, config=vars(args), name=experiment_name, monitor_gym=True, save_code=True)
    writer = SummaryWriter(f"/tmp/{experiment_name}")

# TRY NOT TO MODIFY: seeding
device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')
frame_repeat = 4
frames = 1
env = create_env()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = args.torch_deterministic

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

# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, num_actions, frames=1):
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
            nn.Linear(2560, num_actions)
        )

    def forward(self, x):
        x = torch.Tensor(x).to(device)
        return self.network(x)

def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope =  (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


rb = ReplayBuffer(args.buffer_size)
q_network = QNetwork(env.action_space.n, args.channels).to(device)
target_network = QNetwork(env.action_space.n, args.channels).to(device)
target_network.load_state_dict(q_network.state_dict())
optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
loss_fn = nn.MSELoss()
print(device.__repr__())
print(q_network)
start_time = time.time()

# TRY NOT TO MODIFY: start the game
obs, _ = env.reset(seed=args.seed)
for global_step in range(args.total_timesteps):
    # ALGO LOGIC: put action logic here
    epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction*args.total_timesteps, global_step)
    if random.random() < epsilon:
        action = torch.tensor(random.randint(0, env.action_space.n - 1)).long().item()
    else:
        logits = q_network.forward(obs.reshape((1,)+obs.shape))
        action = torch.argmax(logits, dim=1).tolist()[0]

    # TRY NOT TO MODIFY: execute the game and log data.
    next_obs, reward, done, infos = env.step(action)

    if 'reward' in infos.keys():
        print(f"global_step={global_step}, episodic_return={infos['reward']}")
        writer.add_scalar("charts/episodic_return", infos['reward'], global_step)
    if 'length' in infos.keys():
        writer.add_scalar("charts/episodic_length", infos['length'], global_step)

    rb.put((obs, action, reward, next_obs, done))
    if global_step > args.learning_starts and global_step % args.train_frequency == 0:
        s_obs, s_actions, s_rewards, s_next_obses, s_dones = rb.sample(args.batch_size)
        with torch.no_grad():
            target_max = torch.max(target_network.forward(s_next_obses), dim=1)[0]
            td_target = torch.Tensor(s_rewards).to(device) + args.gamma * target_max * (1 - torch.Tensor(s_dones).to(device))
        old_val = q_network.forward(s_obs).gather(1, torch.LongTensor(s_actions).view(-1,1).to(device)).squeeze()
        loss = loss_fn(td_target, old_val)

        if global_step % 100 == 0:
            writer.add_scalar("losses/td_loss", loss, global_step)
            writer.add_scalar("charts/sps", int(global_step / (time.time() - start_time)), global_step)

        # optimize the midel
        optimizer.zero_grad()
        loss.backward()
        #nn.utils.clip_grad_norm_(list(q_network.parameters()), args.max_grad_norm)
        optimizer.step()

        # update the target network
        if global_step % args.target_network_frequency == 0:
            target_network.load_state_dict(q_network.state_dict())

    # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
    if done:
       obs, _ = env.reset()
    else:
       obs = next_obs


writer.close()
