# https://github.com/pranz24/pytorch-soft-actor-critic

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter
from gym import spaces
import argparse
from distutils.util import strtobool
import collections
import numpy as np
import gym
from gym.wrappers import TimeLimit, Monitor
import pybullet_envs
from gym.spaces import Discrete, Box, MultiBinary, MultiDiscrete, Space
import time
import random
import os
from collections import deque
import cv2
from gym_minigrid.wrappers import *
from vizdoom import DoomGame, Mode, ScreenFormat, ScreenResolution
import skimage.transform
import imageio
from IPython.display import Video
from collections import Counter

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

class WrapFrame(gym.ObservationWrapper):
    def __init__(self, env, width=84, height=84):
        super().__init__(env)
        self.observation_space = env.observation_space.spaces['image']
        self.action_space = Discrete(5)

    def observation(self, obs):
        return obs

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
        vis_obs = obs["image"]

        if done:
            #print(f"rewards: {sum(self._rewards)}")
            info = {"reward": sum(self._rewards),
                    "length": len(self._rewards)}

        return vis_obs, reward, done, info

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SAC with 2 Q functions, Online updates')
    # Common arguments
    parser.add_argument('--exp-name', type=str, default=os.path.basename(__file__).rstrip(".py"),
                        help='the name of this experiment')
    parser.add_argument('--gym-id', type=str, default="MiniGrid-DoorKey-16x16-v0",
                        help='the id of the gym environment')
    parser.add_argument('--learning-rate', type=float, default=3e-4,
                        help='the learning rate of the optimizer')
    parser.add_argument('--seed', type=int, default=9,
                        help='seed of the experiment')
    parser.add_argument('--episode-length', type=int, default=0,
                        help='the maximum length of each episode')
    parser.add_argument('--total-timesteps', type=int, default=4000000,
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
    parser.add_argument('--autotune', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
                        help='automatic tuning of the entropy coefficient.')

    # Algorithm specific arguments
    parser.add_argument('--buffer-size', type=int, default=100000,
                        help='the replay memory buffer size')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='the discount factor gamma')
    parser.add_argument('--target-network-frequency', type=int, default=2, # Denis Yarats' implementation delays this by 2.
                        help="the timesteps it takes to update the target network")
    parser.add_argument('--max-grad-norm', type=float, default=0.5,
                        help='the maximum norm for the gradient clipping')
    parser.add_argument('--batch-size', type=int, default=64, # Worked better in my experiments, still have to do ablation on this. Please remind me
                        help="the batch size of sample from the reply memory")
    parser.add_argument('--tau', type=float, default=0.005,
                        help="target smoothing coefficient (default: 0.005)")
    parser.add_argument('--alpha', type=float, default=0.2,
                        help="Entropy regularization coefficient.")
    parser.add_argument('--learning-starts', type=int, default=1e1,
                        help="timestep to start learning")
    parser.add_argument('--n-step', type=int, default=5,
                        help="n step")


    # Additional hyper parameters for tweaks
    ## Separating the learning rate of the policy and value commonly seen: (Original implementation, Denis Yarats)
    parser.add_argument('--policy-lr', type=float, default=2e-4,
                        help='the learning rate of the policy network optimizer')
    parser.add_argument('--q-lr', type=float, default=2e-4,
                        help='the learning rate of the Q network network optimizer')
    parser.add_argument('--policy-frequency', type=int, default=1,
                        help='delays the update of the actor, as per the TD3 paper.')
    # NN Parameterization
    parser.add_argument('--weights-init', default='xavier', const='xavier', nargs='?', choices=['xavier', "orthogonal", 'uniform'],
                        help='weight initialization scheme for the neural networks.')
    parser.add_argument('--bias-init', default='zeros', const='xavier', nargs='?', choices=['zeros', 'uniform'],
                        help='weight initialization scheme for the neural networks.')

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
env = gym.make(args.gym_id)
env = InfoWrapper(env)
env = WrapFrame(env)
env = wrap_pytorch(env)
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = args.torch_deterministic
env.action_space.seed(args.seed)
env.observation_space.seed(args.seed)
input_shape = env.observation_space.shape
output_shape = env.action_space.shape
# respect the default timelimit

if args.capture_video:
    env = Monitor(env, f'videos/{experiment_name}')

# ALGO LOGIC: initialize agent here:

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Policy(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(Policy, self).__init__()
        self.input_shape = input_shape
        self.num_actions = num_actions

        c, w, h = self.input_shape

        self.conv1 = nn.Conv2d(in_channels=c, out_channels=16, kernel_size=(1, 1), padding=0)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=20, kernel_size=1, padding=0)
        self.fc = nn.Linear(in_features=980, out_features=124)
        self.logits = nn.Linear(in_features=124, out_features=self.num_actions)

        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
                layer.bias.data.zero_()

        nn.init.kaiming_normal_(self.fc.weight, nonlinearity="relu")
        self.fc.bias.data.zero_()
        nn.init.xavier_uniform_(self.logits.weight)
        self.logits.bias.data.zero_()


    def forward(self, x, device):
        x = torch.Tensor(x).to(device)
        x = x
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.contiguous()
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))
        logits = self.logits(x)
        probs = F.softmax(logits, -1)
        z = probs == 0.0
        z = z.float() * 1e-8
        log_probs = torch.log(probs + z)
        return probs, log_probs

    def get_action(self, x, device):
        probs, log_probs = self.forward(x, device)
        dist = Categorical(probs)
        action = dist.sample()
        return action.detach().cpu().numpy()[0], dist

class SoftQNetwork(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(SoftQNetwork, self).__init__()
        self.state_shape = input_shape
        self.n_actions = num_actions

        c, w, h = self.state_shape

        self.conv1 = nn.Conv2d(in_channels=c, out_channels=16, kernel_size=(1, 1), padding=0)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=20, kernel_size=1, padding=0)

        self.fc = nn.Linear(in_features=980, out_features=124)
        self.q_value = nn.Linear(in_features=124, out_features=self.n_actions)

        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
                layer.bias.data.zero_()

        nn.init.kaiming_normal_(self.fc.weight, nonlinearity="relu")
        self.fc.bias.data.zero_()
        nn.init.xavier_uniform_(self.q_value.weight)
        self.q_value.bias.data.zero_()

    def forward(self, x, device):
        x = torch.Tensor(x).to(device)
        x = x
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.contiguous()
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))
        return self.q_value(x)

class ReplayBufferNStep(object):
    def __init__(self, size, n_step, gamma):
        self._storage = deque(maxlen=size)
        self._maxsize = size
        self.n_step_buffer = deque(maxlen=n_step)
        self.gamma = gamma
        self.n_step = n_step

    def __len__(self):
        return len(self._storage)

    def get_n_step(self):
        _, _, reward, next_observation, done = self.n_step_buffer[-1]
        for _, _, r, next_obs, do in reversed(list(self.n_step_buffer)[:-1]):
            reward = self.gamma * reward * (1 - do) + r
            mext_observation, done = (next_obs, do) if do else (next_observation, done)
        return reward, next_observation, done

    def append(self, obs, action, reward, next_obs, done):
        self.n_step_buffer.append((obs, action, reward, next_obs, done))
        if len(self.n_step_buffer) < self.n_step:
            return
        reward, next_obs, done = self.get_n_step()
        obs, action, _, _, _ = self.n_step_buffer[0]
        self._storage.append([obs, action, reward, next_obs, done])

    def sample(self, batch_size):
        idxes = np.random.choice(len(self._storage), batch_size, replace=True)
        obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, action, reward, obs_tp1, done = data
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            dones.append(done)
        return np.array(obses_t), np.array(actions), np.array(rewards), np.array(obses_tp1), np.array(dones)

rb=ReplayBufferNStep(args.buffer_size, args.n_step, args.gamma)

pg = Policy(input_shape, env.action_space.n).to(device)
qf1 = SoftQNetwork(input_shape, env.action_space.n).to(device)
qf2 = SoftQNetwork(input_shape, env.action_space.n).to(device)
qf1_target = SoftQNetwork(input_shape, env.action_space.n).to(device)
qf2_target = SoftQNetwork(input_shape, env.action_space.n).to(device)
qf1_target.eval()
qf2_target.eval()
qf1_target.load_state_dict(qf1.state_dict())
qf2_target.load_state_dict(qf2.state_dict())
values_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
policy_optimizer = optim.Adam(list(pg.parameters()), lr=args.policy_lr)
loss_fn = nn.MSELoss()

# Automatic entropy tuning
if args.autotune:
    target_entropy = 0.98 * (-np.log(1 / env.action_space.n))
    log_alpha = torch.zeros(1, requires_grad=True, device=device)
    alpha = log_alpha.exp().item()
    a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
else:
    alpha = args.alpha

# TRY NOT TO MODIFY: start the game
global_episode = 0
obs, done = env.reset(), False
episode_reward, episode_length= 0.,0
max_episode_reward = -np.inf
for global_step in range(1, args.total_timesteps+1):
    # ALGO LOGIC: put action logic here
    if global_step < args.learning_starts:
        action = env.action_space.sample()
    else:
        action, _ = pg.get_action([obs], device)
    # TRY NOT TO MODIFY: execute the game and log data.
    next_obs, reward, done, _ = env.step(5 if action == 4 else action)

    reward = 100 if reward > 0 else reward
    rb.append(obs, action, reward, next_obs, done)
    episode_reward += reward
    episode_length += 1
    obs = np.array(next_obs)

    # ALGO LOGIC: training.
    if len(rb) > args.batch_size and global_step % 4 == 0: # starts update as soon as there is enough data.
        s_obs, s_actions, s_rewards, s_next_obses, s_dones = rb.sample(args.batch_size)
        with torch.no_grad():
            probs, next_state_log_probs = pg.forward(s_next_obses, device)
            qf1_next_target = qf1_target.forward(s_next_obses, device)
            qf2_next_target = qf2_target.forward(s_next_obses, device)
            min_qf_next_target = (probs * (torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_probs)).sum(-1)
            next_q_value = torch.Tensor(s_rewards).to(device) + (1 - torch.Tensor(s_dones).to(device)) * (args.gamma ** args.n_step) * min_qf_next_target

        qf1_a_values = qf1.forward(s_obs,  device)[np.arange(args.batch_size), np.array(s_actions)]
        qf2_a_values = qf2.forward(s_obs,  device)[np.arange(args.batch_size), np.array(s_actions)]
        qf1_loss = loss_fn(qf1_a_values, next_q_value)
        qf2_loss = loss_fn(qf2_a_values, next_q_value)
        qf_loss = (qf1_loss + qf2_loss) / 2

        values_optimizer.zero_grad()
        qf_loss.backward()
        values_optimizer.step()

        if global_step % args.policy_frequency == 0: # TD 3 Delayed update support
            for _ in range(args.policy_frequency): # compensate for the delay by doing 'actor_update_interval' instead of 1
                probs, log_probs = pg.forward(s_obs, device)
                qf1_pi = qf1.forward(s_obs, device)
                qf2_pi = qf2.forward(s_obs, device)
                min_qf_pi = torch.min(qf1_pi, qf2_pi)
                policy_loss = (probs * (alpha * log_probs - min_qf_pi)).sum(-1).mean()

                policy_optimizer.zero_grad()
                policy_loss.backward()
                policy_optimizer.step()

                if args.autotune:
                    with torch.no_grad():
                        probs, log_probs = pg.forward(s_obs, device)
                    probabilities = (probs * log_probs).sum(-1)
                    alpha_loss = -(log_alpha * (probabilities + target_entropy)).mean()
                    a_optimizer.zero_grad()
                    alpha_loss.backward()
                    a_optimizer.step()
                    alpha = log_alpha.exp().item()

        # update the target network
        if global_step % (args.target_network_frequency) == 0:
            qf1_target.load_state_dict(qf1.state_dict())
            qf1_target.eval()
            qf2_target.load_state_dict(qf2.state_dict())
            qf2_target.eval()
            #for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
            #    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
            #for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
            #    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

    if len(rb) > args.batch_size and global_step % 100 == 0:
        writer.add_scalar("losses/soft_q_value_1_loss", qf1_loss.item(), global_step)
        writer.add_scalar("losses/soft_q_value_2_loss", qf2_loss.item(), global_step)
        writer.add_scalar("losses/qf_loss", qf_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", policy_loss.item(), global_step)
        writer.add_scalar("alpha", alpha, global_step)
        if args.autotune:
            writer.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)

    if done:
        global_episode += 1 # Outside the loop already means the epsiode is done
        writer.add_scalar("charts/episode_reward", episode_reward, global_step)
        writer.add_scalar("charts/episode_length", episode_length, global_step)
        # Terminal verbosity
        if global_episode % 1 == 0:
            print(f"Episode: {global_episode} Step: {global_step}, Ep. Reward: {episode_reward}")

        # Reseting what need to be
        obs, done = env.reset(), False
        episode_reward, episode_length = 0., 0


writer.close()
env.close()
