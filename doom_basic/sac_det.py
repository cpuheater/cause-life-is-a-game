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

class ViZDoomEnv:
    def __init__(self, seed, game_config, render=True, reward_scale=0.1, frame_skip=4):
        # assign observation space
        channel_num = 3

        self.observation_shape = (channel_num, 64, 112)
        self.observation_space = Box(low=0, high=255, shape=self.observation_shape)
        self.reward_scale = reward_scale
        game = DoomGame()

        game.load_config(f"./scenarios/{game_config}.cfg")
        game.set_screen_resolution(ScreenResolution.RES_160X120)
        game.set_screen_format(ScreenFormat.CRCGCB)

        num_buttons = game.get_available_buttons_size()
        self.action_space = Discrete(num_buttons)
        actions = [([False] * num_buttons) for i in range(num_buttons)]
        for i in range(num_buttons):
            actions[i][i] = True
        self.actions = actions
        self.frame_skip = frame_skip

        game.set_seed(seed)
        game.set_window_visible(render)
        game.init()

        self.game = game

    def get_current_input(self):
        state = self.game.get_state()
        res_source = []
        res_source.append(state.screen_buffer)
        res = np.vstack(res_source)
        res = skimage.transform.resize(res, self.observation_space.shape, preserve_range=True)
        self.last_input = res
        return res

    def step(self, action):
        info = {}
        reward = self.game.make_action(self.actions[action], self.frame_skip)
        done = self.game.is_episode_finished()
        if done:
            ob = self.last_input
        else:
            ob = self.get_current_input()
        # reward scaling
        reward = reward * self.reward_scale
        self.total_reward += reward
        self.total_length += 1

        if done:
            info['Episode_Total_Reward'] = self.total_reward
            info['Episode_Total_Len'] = self.total_length

        return ob, reward, done, info

    def reset(self):
        self.game.new_episode()
        self.total_reward = 0
        self.total_length = 0
        ob = self.get_current_input()
        return ob

    def close(self):
        self.game.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SAC with 2 Q functions, Online updates')
    # Common arguments
    parser.add_argument('--exp-name', type=str, default=os.path.basename(__file__).rstrip(".py"),
                        help='the name of this experiment')
    parser.add_argument('--gym-id', type=str, default="basic",
                        help='the id of the gym environment')
    parser.add_argument('--learning-rate', type=float, default=3e-4,
                        help='the learning rate of the optimizer')
    parser.add_argument('--seed', type=int, default=2,
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
    parser.add_argument('--autotune', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
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
    parser.add_argument('--learning-starts', type=int, default=5e1,
                        help="timestep to start learning")


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
env = ViZDoomEnv(args.seed, args.gym_id, render=True, reward_scale=0.01, frame_skip=4)
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
LOG_STD_MAX = 2
LOG_STD_MIN = -5

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Policy(nn.Module):
    def __init__(self, num_actions):
        super(Policy, self).__init__()
        self.num_actions = num_actions

        self.network = nn.Sequential(
            layer_init(nn.Conv2d(3, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(2560, 512)),
            nn.ReLU()
        )
        self.logits = nn.Linear(in_features=512, out_features=self.num_actions)

        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
                layer.bias.data.zero_()

        nn.init.xavier_uniform_(self.logits.weight)
        self.logits.bias.data.zero_()


    def forward(self, x, device):
        x = torch.Tensor(x).to(device)
        x = x / 255.0
        x = self.network(x)
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
    def __init__(self, num_actions):
        super(SoftQNetwork, self).__init__()
        self.n_actions = num_actions

        self.network = nn.Sequential(
            layer_init(nn.Conv2d(3, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(2560, 512)),
            nn.ReLU()
        )

        self.q_value = nn.Linear(in_features=512, out_features=num_actions)

        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
                layer.bias.data.zero_()

        nn.init.xavier_uniform_(self.q_value.weight)
        self.q_value.bias.data.zero_()

    def forward(self, x, device):
        x = torch.Tensor(x).to(device)
        x = x / 255.0
        x = self.network(x)
        return self.q_value(x)


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

rb = ReplayBuffer(args.buffer_size)
pg = Policy(env.action_space.n).to(device)
qf1 = SoftQNetwork(env.action_space.n).to(device)
qf2 = SoftQNetwork(env.action_space.n).to(device)
qf1_target = SoftQNetwork(env.action_space.n).to(device)
qf2_target = SoftQNetwork(env.action_space.n).to(device)
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
    next_obs, reward, done, _ = env.step(action)
    #if not done:
    #    if reward < 0:
    #        reward = -1

    reward = 100 if reward > 0 else reward
    rb.put((obs, action, reward, next_obs, done))
    episode_reward += reward
    episode_length += 1
    obs = np.array(next_obs)

    # ALGO LOGIC: training.
    if len(rb.buffer) > args.batch_size and global_step % 4 == 0: # starts update as soon as there is enough data.
        s_obs, s_actions, s_rewards, s_next_obses, s_dones = rb.sample(args.batch_size)
        with torch.no_grad():
            probs, next_state_log_probs = pg.forward(s_next_obses, device)
            qf1_next_target = qf1_target.forward(s_next_obses, device)
            qf2_next_target = qf2_target.forward(s_next_obses, device)
            min_qf_next_target = (probs * (torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_probs)).sum(-1)
            next_q_value = torch.Tensor(s_rewards).to(device) + (1 - torch.Tensor(s_dones).to(device)) * args.gamma * min_qf_next_target

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

    if len(rb.buffer) > args.batch_size and global_step % 100 == 0:
        writer.add_scalar("losses/soft_q_value_1_loss", qf1_loss.item(), global_step)
        writer.add_scalar("losses/soft_q_value_2_loss", qf2_loss.item(), global_step)
        writer.add_scalar("losses/qf_loss", qf_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", policy_loss.item(), global_step)
        writer.add_scalar("alpha", alpha, global_step)
        if args.autotune:
            writer.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)

    if done:
        print(f"Episode reward {episode_reward}")
        global_episode += 1 # Outside the loop already means the epsiode is done
        writer.add_scalar("charts/episode_reward", episode_reward, global_step)
        writer.add_scalar("charts/episode_length", episode_length, global_step)
        # Terminal verbosity
        if global_episode % 10 == 0:
            print(f"Episode: {global_episode} Step: {global_step}, Ep. Reward: {episode_reward}")

        # Reseting what need to be
        obs, done = env.reset(), False
        episode_reward, episode_length = 0., 0

writer.close()
env.close()
