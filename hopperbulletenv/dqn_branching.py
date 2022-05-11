# Reference: https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import pybullet_envs
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DQN agent')
    # Common arguments
    parser.add_argument('--exp-name', type=str, default=os.path.basename(__file__).rstrip(".py"),
                        help='the name of this experiment')
    parser.add_argument('--gym-id', type=str, default="Hopper-v2",
                        help='the id of the gym environment')
    parser.add_argument('--learning-rate', type=float, default=7e-4,
                        help='the learning rate of the optimizer')
    parser.add_argument('--seed', type=int, default=1,
                        help='seed of the experiment')
    parser.add_argument('--total-timesteps', type=int, default=10000000,
                        help='total timesteps of the experiments')
    parser.add_argument('--torch-deterministic', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
                        help='if toggled, `torch.backends.cudnn.deterministic=False`')
    parser.add_argument('--cuda', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
                        help='if toggled, cuda will not be enabled by default')
    parser.add_argument('--track', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
                        help='run the script in production mode and use wandb to log outputs')
    parser.add_argument('--capture-video', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
                        help='weather to capture videos of the agent performances (check out `videos` folder)')
    parser.add_argument('--wandb-project-name', type=str, default="cleanRL",
                        help="the wandb's project name")
    parser.add_argument('--wandb-entity', type=str, default=None,
                        help="the entity (team) of wandb's project")

    # Algorithm specific arguments
    parser.add_argument('--buffer-size', type=int, default=10000,
                        help='the replay memory buffer size')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='the discount factor gamma')
    parser.add_argument('--target-network-frequency', type=int, default=500,
                        help="the timesteps it takes to update the target network")
    parser.add_argument('--max-grad-norm', type=float, default=0.5,
                        help='the maximum norm for the gradient clipping')
    parser.add_argument('--batch-size', type=int, default=32,
                        help="the batch size of sample from the reply memory")
    parser.add_argument('--start-e', type=float, default=1,
                        help="the starting epsilon for exploration")
    parser.add_argument('--end-e', type=float, default=0.05,
                        help="the ending epsilon for exploration")
    parser.add_argument('--exploration-fraction', type=float, default=0.8,
                        help="the fraction of `total-timesteps` it takes from start-e to go end-e")
    parser.add_argument('--learning-starts', type=int, default=32,
                        help="timestep to start learning")
    parser.add_argument('--train-frequency', type=int, default=1,
                        help="the frequency of training")
    parser.add_argument('--bins', type=int, default=6,
                        help="number of bins")
    args = parser.parse_args()
    if not args.seed:
        args.seed = int(time.time())

def one_hot(a, size):
    b = np.zeros((size))
    b[a] = 1
    return b

# TRY NOT TO MODIFY: setup the environment
experiment_name = f"{args.gym_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
writer = SummaryWriter(f"runs/{experiment_name}")
writer.add_text('hyperparameters', "|param|value|\n|-|-|\n%s" % (
    '\n'.join([f"|{key}|{value}|" for key, value in vars(args).items()])))
if args.track:
    import wandb
    wandb.init(project=args.wandb_project_name, entity=args.wandb_entity, sync_tensorboard=True, config=vars(args), name=experiment_name, monitor_gym=True, save_code=True)
    writer = SummaryWriter(f"/tmp/{experiment_name}")


# TRY NOT TO MODIFY: seeding
device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')
env = gym.make(args.gym_id)
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = args.torch_deterministic
env.seed(args.seed)
env.action_space.seed(args.seed)
env.observation_space.seed(args.seed)
# respect the default timelimit
#assert isinstance(env.action_space, Continous), "only discrete action space is supported"
if args.capture_video:
    env = Monitor(env, f'videos/{experiment_name}')

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

# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, env, num_bins=6):
        super(QNetwork, self).__init__()
        self.num_bins = num_bins
        self.network = nn.Sequential(nn.Linear(np.array(env.observation_space.shape).prod(), 128),
                                     nn.ReLU(),
                                     nn.Linear(128, 128),
                                     nn.ReLU())
        self.v = nn.Linear(128, 1)
        self.a_heads = nn.ModuleList([nn.Linear(128, num_bins) for i in range(env.action_space.shape[0])])
        #self.a = nn.Linear(128, num_bins * env.action_space.shape[0])

    def forward(self, x, device):
        x = torch.Tensor(x).to(device)
        x = self.network(x)
        v = self.v(x)
        a = torch.stack([h(x) for h in self.a_heads], dim = 1)
        tmp_a = [h(x) for h in self.a_heads]
        q = v.unsqueeze(2) + a - a.mean(2, keepdim = True )
        tmp_q = [v + a - a.mean(1, keepdim=True) for a in tmp_a]
        q = q.view(x.shape[0], -1)
        q = torch.split(q, [self.num_bins ]*env.action_space.shape[0], dim=1)
        return tmp_q

def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope =  (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)

rb = ReplayBuffer(args.buffer_size)
q_network = QNetwork(env, args.bins).to(device)
target_network = QNetwork(env, args.bins).to(device)
target_network.load_state_dict(q_network.state_dict())
optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
loss_fn = nn.MSELoss()
print(device.__repr__())
print(q_network)

# TRY NOT TO MODIFY: start the game
obs = env.reset()
episode_reward = 0
bins = np.linspace(-1.,1., args.bins)
episode = 0
for global_step in range(args.total_timesteps):
    # ALGO LOGIC: put action logic here
    epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction*args.total_timesteps, global_step)
    #if random.random() < epsilon:
    #    action = env.action_space.sample()
    #else:
    logits = q_network.forward(obs.reshape((1,)+obs.shape), device)
    action = [int(torch.argmax(l.squeeze(0)).cpu().numpy()) for l in logits]

    action_cont = np.array([bins[aa] for aa in action])

    # TRY NOT TO MODIFY: execute the game and log data.
    next_obs, reward, done, _ = env.step(action_cont)
    episode_reward += reward

    # ALGO LOGIC: training.
    rb.put((obs, action, reward, next_obs, done))
    if global_step > args.learning_starts and global_step % args.train_frequency == 0:
        s_obs, s_actions, s_rewards, s_next_obses, s_dones = rb.sample(args.batch_size)
        with torch.no_grad():
            max_action = [torch.argmax(q, dim=1) for q in q_network.forward(s_next_obses, device)]
            target_q_next = target_network.forward(s_next_obses, device)
            target_max = [q.gather(1, m_a.unsqueeze(1)) for q, m_a in zip(target_q_next, max_action)]
            target_q = torch.stack([torch.Tensor(s_rewards).to(device) + args.gamma * t_m.squeeze(1) * (1 - torch.Tensor(s_dones).to(device)) for t_m in target_max]).T
        curr_q = torch.stack([q.gather(1, a.unsqueeze(1)) for q, a in zip(q_network.forward(s_obs, device), torch.LongTensor(s_actions).to(device).T)]).squeeze(2).T
        loss = loss_fn(target_q, curr_q)

        if global_step % 100 == 0:
            writer.add_scalar("losses/td_loss", loss, global_step)

        # optimize the midel
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(list(q_network.parameters()), args.max_grad_norm)
        optimizer.step()

        # update the target network
        if global_step % args.target_network_frequency == 0:
            target_network.load_state_dict(q_network.state_dict())

    # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
    obs = next_obs

    if done:
        # TRY NOT TO MODIFY: record rewards for plotting purposes
        episode += 1
        print(f"global_step={global_step}, episode_reward={episode_reward}")
        writer.add_scalar("charts/episodic_return_episode", episode_reward, episode)
        writer.add_scalar("charts/episodic_return", episode_reward, global_step)
        writer.add_scalar("charts/epsilon", epsilon, global_step)
        obs, episode_reward = env.reset(), 0

env.close()
writer.close()
