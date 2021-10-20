import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

import argparse
from distutils.util import strtobool
import numpy as np
import gym
from gym.wrappers import TimeLimit, Monitor
from gym.spaces import Discrete, Box, MultiBinary, MultiDiscrete, Space
import time
import random
import os
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnvWrapper
from dataclasses import dataclass
import math

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PPO agent')
    # Common arguments
    parser.add_argument('--exp-name', type=str, default=os.path.basename(__file__).rstrip(".py"),
                        help='the name of this experiment')
    parser.add_argument('--gym-id', type=str, default="CartPole-v0",
                        help='the id of the gym environment')
    parser.add_argument('--learning-rate', type=float, default=3e-4,
                        help='the learning rate of the optimizer')
    parser.add_argument('--seed', type=int, default=1,
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

    # Algorithm specific arguments
    parser.add_argument('--n-minibatch', type=int, default=4,
                        help='the number of mini batch')
    parser.add_argument('--num-envs', type=int, default=32,
                        help='the number of parallel game environment')
    parser.add_argument('--num-steps', type=int, default=512,
                        help='the number of steps per game environment')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='the discount factor gamma')
    parser.add_argument('--gae-lambda', type=float, default=0.95,
                        help='the lambda for the general advantage estimation')
    parser.add_argument('--ent-coef', type=float, default=0.01,
                        help="coefficient of the entropy")
    parser.add_argument('--vf-coef', type=float, default=0.5,
                        help="coefficient of the value function")
    parser.add_argument('--max-grad-norm', type=float, default=0.5,
                        help='the maximum norm for the gradient clipping')
    parser.add_argument('--clip-coef', type=float, default=0.2,
                        help="the surrogate clipping coefficient")
    parser.add_argument('--update-epochs', type=int, default=4,
                         help="the K epochs to update the policy")
    parser.add_argument('--kle-stop', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
                         help='If toggled, the policy updates will be early stopped w.r.t target-kl')
    parser.add_argument('--kle-rollback', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
                         help='If toggled, the policy updates will roll back to previous policy if KL exceeds target-kl')
    parser.add_argument('--target-kl', type=float, default=0.03,
                         help='the target-kl variable that is referred by --kl')
    parser.add_argument('--gae', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
                         help='Use GAE for advantage computation')
    parser.add_argument('--norm-adv', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
                          help="Toggles advantages normalization")
    parser.add_argument('--anneal-lr', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
                          help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument('--clip-vloss', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
                          help='Toggles wheter or not to use a clipped loss for the value function, as per the paper.')
    parser.add_argument('--rnn-hidden-size', type=int, default=128,
                        help='rnn hidden size')
    parser.add_argument('--rnn-seq-len', type=int, default=8,
                        help='rnn seq length')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='batch size')


    args = parser.parse_args()
    if not args.seed:
        args.seed = int(time.time())

args.batch_size = int(args.num_envs * args.num_steps)
args.minibatch_size = int(args.batch_size // args.n_minibatch)

class VecPyTorch(VecEnvWrapper):
    def __init__(self, venv, device):
        super(VecPyTorch, self).__init__(venv)
        self.device = device

    def reset(self):
        obs = self.venv.reset()
        obs = torch.from_numpy(obs).float().to(self.device)
        return obs

    def step_async(self, actions):
        actions = actions.cpu().numpy()
        self.venv.step_async(actions)

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        obs = torch.from_numpy(obs).float().to(self.device)
        reward = torch.from_numpy(reward).unsqueeze(dim=1).float()
        return obs, reward, done, info

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
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = args.torch_deterministic
def make_env(gym_id, seed, idx):
    def thunk():
        env = gym.make(gym_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if args.capture_video:
            if idx == 0:
                env = Monitor(env, f'videos/{experiment_name}')
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env
    return thunk
envs = VecPyTorch(DummyVecEnv([make_env(args.gym_id, args.seed+i, i) for i in range(args.num_envs)]), device)
# if args.prod_mode:
#     envs = VecPyTorch(
#         SubprocVecEnv([make_env(args.gym_id, args.seed+i, i) for i in range(args.num_envs)], "fork"),
#         device
#     )
assert isinstance(envs.action_space, Discrete), "only discrete action space is supported"

def calc_discounted_return(rewards, discount, final_value):
    # Calculate discounted returns based on rewards and discount factor
    seq_len = len(rewards)
    discounted_returns = torch.zeros(seq_len)
    discounted_returns[-1] = rewards[-1] + discount * final_value
    for i in range(seq_len - 2, -1, -1):
        discounted_returns[i] = rewards[i] + discount * discounted_returns[i + 1]
    return discounted_returns


def compute_advantages(rewards, values, discount, gae_lambda):
    # Compute General Advantage.
    deltas = rewards + discount * values[1:] - values[:-1]
    seq_len = len(rewards)
    advs = torch.zeros(seq_len + 1)
    multiplier = discount * gae_lambda
    for i in range(seq_len - 1, -1, -1):
        advs[i] = advs[i + 1] * multiplier + deltas[i]
    return advs[:-1]


@dataclass
class TrajectorBatch():
    # Dataclass for storing data batch
    obs: torch.tensor
    actions: torch.tensor
    logprobs: torch.tensor
    advantages: torch.tensor
    discounted_returns: torch.tensor
    batch_size: torch.tensor
    rnn_hidden_states: torch.tensor
    rnn_cell_states: torch.tensor


class TrajectoryDataset():
    # Dataset for producing training batches from trajectories

    def __init__(self, trajectories, batch_size, device, batch_len):

        # Combine multiple trajectories into
        self.trajectories = {key: value.to(device) for key, value in trajectories.items()}
        self.batch_len = batch_len
        truncated_seq_len = torch.clamp(trajectories["seq_len"] - batch_len + 1, 0, args.num_steps)
        self.cumsum_seq_len = np.cumsum(np.concatenate((np.array([0]), truncated_seq_len.numpy())))
        self.batch_size = batch_size

    def __iter__(self):
        self.valid_idx = np.arange(self.cumsum_seq_len[-1])
        self.batch_count = 0
        return self

    def __next__(self):
        if self.batch_count * self.batch_size >= math.ceil(self.cumsum_seq_len[-1] / self.batch_len):
            raise StopIteration
        else:
            actual_batch_size = min(len(self.valid_idx), self.batch_size)
            start_idx = np.random.choice(self.valid_idx, size=actual_batch_size, replace=False)
            self.valid_idx = np.setdiff1d(self.valid_idx, start_idx)
            eps_idx = np.digitize(start_idx, bins=self.cumsum_seq_len, right=False) - 1
            seq_idx = start_idx - self.cumsum_seq_len[eps_idx]
            series_idx = np.linspace(seq_idx, seq_idx + self.batch_len - 1, num=self.batch_len, dtype=np.int64)
            self.batch_count += 1
            return TrajectorBatch(**{key: value[eps_idx, series_idx] for key, value
                                     in self.trajectories.items() if key in TrajectorBatch.__dataclass_fields__.keys()},
                                  batch_size=actual_batch_size)

def split_trajectories_episodes(trajectory_tensors):
    states_episodes, actions_episodes, action_probabilities_episodes = [], [], []
    rewards_episodes, terminal_rewards_episodes, terminals_episodes, values_episodes = [], [], [], []
    policy_hidden_episodes, policy_cell_episodes, critic_hidden_episodes, critic_cell_episodes = [], [], [], []
    len_episodes = []
    trajectory_episodes = {key: [] for key in trajectory_tensors.keys()}
    for i in range(args.num_envs):
        terminals_tmp = trajectory_tensors["dones"].clone()
        terminals_tmp[0, i] = 1
        terminals_tmp[-1, i] = 1
        split_points = (terminals_tmp[:, i] == 1).nonzero() + 1

        split_lens = split_points[1:] - split_points[:-1]
        split_lens[0] += 1

        len_episode = [split_len.item() for split_len in split_lens]
        len_episodes += len_episode
        for key, value in trajectory_tensors.items():
            # Value includes additional step
            if key == "values":
                value_split = list(torch.split(value[:, i], len_episode[:-1] + [len_episode[-1] + 1]))
                # Append extra 0 to values to represent no future reward
                for j in range(len(value_split) - 1):
                    value_split[j] = torch.cat((value_split[j], torch.zeros(1).to(device)))
                trajectory_episodes[key] += value_split
            else:
                trajectory_episodes[key] += torch.split(value[:, i], len_episode)
    return trajectory_episodes, len_episodes


def pad_and_compute_returns(trajectory_episodes, len_episodes):
    # Pad the trajectories up to hp.rollout_steps so they can be combined in a single tensor
    # Add advantages and discounted_returns to trajectories

    episode_count = len(len_episodes)
    advantages_episodes, discounted_returns_episodes = [], []
    padded_trajectories = {key: [] for key in trajectory_episodes.keys()}
    padded_trajectories["advantages"] = []
    padded_trajectories["discounted_returns"] = []

    for i in range(episode_count):
        single_padding = torch.zeros(args.num_steps - len_episodes[i])
        for key, value in trajectory_episodes.items():
            if value[i].ndim > 1:
                padding = torch.zeros(args.num_steps - len_episodes[i], value[0].shape[1], dtype=value[i].dtype).to(device)
            else:
                padding = torch.zeros(args.num_steps - len_episodes[i], dtype=value[i].dtype).to(device)
            padded_trajectories[key].append(torch.cat((value[i], padding)))
        padded_trajectories["advantages"].append(
            torch.cat((compute_advantages(rewards=trajectory_episodes["rewards"][i],
                                          values=trajectory_episodes["values"][i],
                                          discount=args.gamma,
                                          gae_lambda=args.gae_lambda), single_padding)))
        padded_trajectories["discounted_returns"].append(
            torch.cat((calc_discounted_return(rewards=trajectory_episodes["rewards"][i],
                                              discount=args.gae_lambda,
                                              final_value=trajectory_episodes["values"][i][-1]), single_padding)))
    return_val = {k: torch.stack(v) for k, v in padded_trajectories.items()}
    return_val["seq_len"] = torch.tensor(len_episodes)

    return return_val


# ALGO LOGIC: initialize agent here:
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

class Agent(nn.Module):
    def __init__(self, envs):
        super(Agent, self).__init__()
        self.network = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.observation_space.shape).prod(), 128)),
            nn.ReLU(),
            layer_init(nn.Linear(128, args.rnn_hidden_size)),
            nn.ReLU()
        )
        self.lstm = nn.LSTM(args.rnn_hidden_size, args.rnn_hidden_size)
        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                nn.init.orthogonal_(param, np.sqrt(2))
        self.actor = layer_init(nn.Linear(args.rnn_hidden_size, envs.action_space.n), std=0.01)
        self.critic = layer_init(nn.Linear(args.rnn_hidden_size, 1), std=1)

    def forward(self, x, hxs, mask):
        x = self.network(x)
        hs = hxs[0]
        cs = hxs[1]
        if mask is not None:
            hs, cs = (hs * mask), (cs * mask)
        x, (hs, cs) = self.lstm(x.unsqueeze(0), (hs.unsqueeze(0), cs.unsqueeze(0)))
        x = x.squeeze()
        hs = hs.squeeze(0)
        cs = cs.squeeze(0)
        hxs = torch.stack([hs, cs])
        return x, hxs

    def get_action(self, x, rnn_hidden_state, mask, action=None):
        x, rnn_hidden_state = self.forward(x, rnn_hidden_state, mask)
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return rnn_hidden_state, action, probs.log_prob(action), probs.entropy()

    def get_value(self, x, rnn_hidden_state, mask):
        x, rnn_hidden_state = self.forward(x, rnn_hidden_state, mask)
        return self.critic(x)

agent = Agent(envs).to(device)
optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
if args.anneal_lr:
    # https://github.com/openai/baselines/blob/ea25b9e8b234e6ee1bca43083f8f3cf974143998/baselines/ppo2/defaults.py#L20
    lr = lambda f: f * args.learning_rate

# ALGO Logic: Storage for epoch data
obs = torch.zeros((args.num_steps, args.num_envs) + envs.observation_space.shape).to(device)
actions = torch.zeros((args.num_steps, args.num_envs) + envs.action_space.shape).to(device)
logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
values = torch.zeros((args.num_steps+1, args.num_envs)).to(device)
rnn_hidden_states = torch.zeros((args.num_steps, 2, args.num_envs, args.rnn_hidden_size)).to(device)
masks = torch.ones((args.num_steps, args.num_envs, 1)).to(device)

# TRY NOT TO MODIFY: start the game
global_step = 0
# Note how `next_obs` and `next_done` are used; their usage is equivalent to
# https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/84a7582477fb0d5c82ad6d850fe476829dddd2e1/a2c_ppo_acktr/storage.py#L60
next_obs = envs.reset()
next_done = torch.zeros(args.num_envs).to(device)
num_updates = args.total_timesteps // args.batch_size
mask = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in next_done]).to(device)
rnn_hidden_state = torch.zeros((2, args.num_envs, args.rnn_hidden_size)).to(device)
for update in range(1, num_updates+1):
    # Annealing the rate if instructed to do so.
    if args.anneal_lr:
        frac = 1.0 - (update - 1.0) / num_updates
        lrnow = lr(frac)
        optimizer.param_groups[0]['lr'] = lrnow
    mask = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in next_done]).to(device)
    rnn_hidden_state = torch.zeros((2, args.num_envs, args.rnn_hidden_size)).to(device)

    # TRY NOT TO MODIFY: prepare the execution of the game.
    for step in range(0, args.num_steps):
        global_step += 1 * args.num_envs
        obs[step] = next_obs
        dones[step] = next_done
        rnn_hidden_states[step] = rnn_hidden_state
        masks[step] = mask

        # ALGO LOGIC: put action logic here
        with torch.no_grad():
            values[step] = agent.get_value(obs[step], rnn_hidden_state, mask).flatten()
            rnn_hidden_state, action, logproba, _ = agent.get_action(obs[step], rnn_hidden_state, mask)

        actions[step] = action
        logprobs[step] = logproba

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rs, ds, infos = envs.step(action)
        rewards[step], next_done = rs.view(-1), torch.Tensor(ds).to(device)
        mask = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in next_done]).to(device)

        for info in infos:
            if 'episode' in info.keys():
                print(f"global_step={global_step}, episode_reward={info['episode']['r']}")
                writer.add_scalar("charts/episode_reward", info['episode']['r'], global_step)
                break

    # bootstrap reward if not done. reached the batch limit
    with torch.no_grad():
        last_value = agent.get_value(next_obs.to(device), rnn_hidden_state, mask).reshape(1, -1)
    values[-1] = last_value * (1 - next_done)
    trajectories = {"obs": obs,
                       "actions": actions,
                       "logprobs": logprobs,
                       "rewards": rewards,
                       "values": values,
                       "dones": dones,
                       "rnn_hidden_states": rnn_hidden_states[:, 0, :],
                       "cell_hidden_states": rnn_hidden_states[:, 1, :]
                    }

    trajectory_episodes, len_episodes = split_trajectories_episodes(trajectories)
    trajectories = pad_and_compute_returns(trajectory_episodes, len_episodes)

    trajectory_dataset = TrajectoryDataset(trajectories, batch_size=args.batch_size,
                                           device=device, batch_len=args.rnn_seq_len)

    # Optimizaing the policy and value network
    for i_epoch_pi in range(args.update_epochs):
        for batch in trajectory_dataset:
            #if args.norm_adv:
            #    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

            _, _, newlogproba, entropy = agent.get_action(batch.obs, rnn_hidden_state[:1], None, batch.actions.long())
            ratio = (newlogproba - batch.action_probabilities[-1, :]).exp()

            # Stats
            approx_kl = (batch.action_probabilities[-1, :] - newlogproba).mean()

            # Policy loss
            pg_loss1 = batch.advantages[-1, :] * ratio
            pg_loss2 = batch.advantages[-1, :] * torch.clamp(ratio, 1-args.clip_coef, 1+args.clip_coef)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()
            entropy_loss = entropy.mean()

            # Value loss
            new_values = agent.get_value(batch.obs, rnn_hidden_state[:1], None).view(-1)
            #if args.clip_vloss:
             #   v_loss_unclipped = ((new_values - b_returns[minibatch_ind]) ** 2)
             #   v_clipped = b_values[minibatch_ind] + torch.clamp(new_values - b_values[minibatch_ind], -args.clip_coef, args.clip_coef)
             #   v_loss_clipped = (v_clipped - b_returns[minibatch_ind])**2
             #   v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
             #   v_loss = 0.5 * v_loss_max.mean()
            #else:
            v_loss = 0.5 * ((new_values - batch.discounted_returns[-1, :]) ** 2).mean()

            loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
            optimizer.step()


    # TRY NOT TO MODIFY: record rewards for plotting purposes
    writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]['lr'], global_step)
    writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
    writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
    writer.add_scalar("losses/entropy", entropy.mean().item(), global_step)
    writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
    if args.kle_stop or args.kle_rollback:
        writer.add_scalar("debug/pg_stop_iter", i_epoch_pi, global_step)

envs.close()
writer.close()
