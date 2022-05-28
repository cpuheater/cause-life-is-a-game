# Reference: https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf

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
import gym
import gym_microrts
from gym.wrappers import TimeLimit, Monitor
from gym_microrts.envs.vec_env import MicroRTSGridModeVecEnv
from gym_microrts import microrts_ai
from gym.spaces import MultiDiscrete
import time
import random
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DQN agent')
    # Common arguments
    parser.add_argument('--exp-name', type=str, default=os.path.basename(__file__).rstrip(".py"),
                        help='the name of this experiment')
    parser.add_argument('--gym-id', type=str, default="Microrts4-randomBiasedAI",
                        help='the id of the gym environment')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
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
    parser.add_argument('--num-bot-envs', type=int, default=1,
                        help='the number of bot game environment; 16 bot envs measn 16 games')
    parser.add_argument('--num-selfplay-envs', type=int, default=0,
                        help='the number of self play envs; 16 self play envs means 8 games')

    # Algorithm specific arguments
    parser.add_argument('--buffer-size', type=int, default=100000,
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
    #if not args.seed:
    args.seed = int(time.time())

# TRY NOT TO MODIFY: setup the environment
experiment_name = f"{args.gym_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
writer = SummaryWriter(f"runs/{experiment_name}")
writer.add_text('hyperparameters', "|param|value|\n|-|-|\n%s" % (
    '\n'.join([f"|{key}|{value}|" for key, value in vars(args).items()])))
if args.track:
    import wandb
    wandb.init(project=args.wandb_project_name, entity=args.wandb_entity, sync_tensorboard=True, config=vars(args), name=experiment_name, monitor_gym=True, save_code=True)
    writer = SummaryWriter(f"/tmp/{experiment_name}")


env = MicroRTSGridModeVecEnv(
    num_selfplay_envs=args.num_selfplay_envs,
    num_bot_envs=args.num_bot_envs,
    max_steps=1200,
    render_theme=2,
    ai2s=[microrts_ai.randomBiasedAI for _ in range(args.num_bot_envs)],
    map_paths=["maps/8x8/basesWorkers8x8.xml"],
    reward_weight=np.array([10.0, 1.0, 1.0, 0.2, 1.0, 4.0])
)


# TRY NOT TO MODIFY: seeding
device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = args.torch_deterministic
env.action_space.seed(args.seed)
env.observation_space.seed(args.seed)
mapsize = 8 * 8
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
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst, invalid_mask_lst = [], [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done_mask, invalid_mask = transition
            s_lst.append(s)
            a_lst.append(a)
            r_lst.append(r)
            s_prime_lst.append(s_prime)
            done_mask_lst.append(done_mask)
            invalid_mask_lst.append(invalid_mask)

        return np.array(s_lst), np.array(a_lst), \
               np.array(r_lst), np.array(s_prime_lst), \
               np.array(done_mask_lst), \
               torch.stack(invalid_mask_lst)

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, env):
        super(QNetwork, self).__init__()
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(27, 16, kernel_size=3, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(16, 32, kernel_size=2)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(128, 128)),
            nn.ReLU())

        self.v = nn.Linear(128, 1)
        self.a_heads = nn.ModuleList([nn.Linear(128, mapsize * n) for n in env.action_plane_space.nvec.tolist()])

    def forward(self, x, device):
        x = torch.Tensor(x).to(device)
        x = self.network(x.permute((0, 3, 1, 2)))
        v = self.v(x)
        a_list = [h(x) for h in self.a_heads]
        q = [v.repeat(1, 64).unsqueeze(2) + a.view(x.shape[0], mapsize, -1) - a.view(x.shape[0], mapsize, -1).mean(-1, keepdim=True) for a in a_list]
        return q

def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope =  (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)

rb = ReplayBuffer(args.buffer_size)
q_network = QNetwork(env).to(device)
target_network = QNetwork(env).to(device)
target_network.load_state_dict(q_network.state_dict())
optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
loss_fn = nn.MSELoss()
print(device.__repr__())
print(q_network)

# TRY NOT TO MODIFY: start the game
obs = env.reset()
episode_reward = 0
episode = 0
for global_step in range(args.total_timesteps):
    # ALGO LOGIC: put action logic here
    env.render()
    epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction*args.total_timesteps, global_step)
    invalid_action_masks = torch.tensor(np.array(env.get_action_mask())).to(device)
    if random.random() < epsilon:
        logits = torch.randn(invalid_action_masks.shape).to(device).squeeze(0)
        logits = torch.split(logits, env.action_plane_space.nvec.tolist(), dim=1)
    else:
        logits = q_network.forward(obs, device)
    split_invalid_action_masks = torch.split(invalid_action_masks.view(-1, env.action_plane_space.nvec.sum()), env.action_plane_space.nvec.tolist(), dim=1)
    logits_masked = [torch.where(masks.bool(), l.squeeze(0), torch.tensor(float("-inf")).to(device)) for l, masks in zip(logits, split_invalid_action_masks)]
    action = np.array([torch.argmax(l, dim=1).cpu().numpy() for l in logits_masked]).transpose()

    # TRY NOT TO MODIFY: execute the game and log data.
    try:
        next_obs, reward, done, infos = env.step(action.reshape(1, -1))
    except Exception as e:
        e.printStackTrace()
        raise

    episode_reward += reward

    # ALGO LOGIC: training.
    rb.put((obs.squeeze(0), action, reward, next_obs.squeeze(0), done, invalid_action_masks.squeeze(0)))
    if global_step > args.learning_starts and global_step % args.train_frequency == 0:
        s_obs, s_actions, s_rewards, s_next_obses, s_dones, invalid_action_masks = rb.sample(args.batch_size)
        with torch.no_grad():
            split_invalid_action_masks = torch.split(invalid_action_masks.view(-1, env.action_plane_space.nvec.sum()), env.action_plane_space.nvec.tolist(), dim=1)
            q = q_network.forward(s_next_obses, device)
            max_action = [torch.argmax(torch.where(masks.view_as(q).bool(), q, torch.tensor(float("-inf")).to(device)), dim=2) for q, masks in zip(q, split_invalid_action_masks)]
            target_q_next = target_network.forward(s_next_obses, device)
            target_max = [q.gather(1, m_a.unsqueeze(2)) for q, m_a in zip(target_q_next, max_action)]
            target_q = torch.stack([torch.Tensor(s_rewards).to(device).repeat(1, 64) + args.gamma * t_m.squeeze(2) * ((1 - torch.Tensor(s_dones).to(device)).repeat(1, 64)) for t_m in target_max]).view(-1, mapsize, 7)
        curr_q = torch.stack([q.gather(2, a.unsqueeze(1)) for q, a in zip(q_network.forward(s_obs, device), torch.LongTensor(s_actions).to(device).view(-1, 7).T.view(7, -1, mapsize))]).unsqueeze(2).view(-1, mapsize, 7)
        loss = loss_fn(target_q.view(-1, 7), curr_q.view(-1, 7))

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
