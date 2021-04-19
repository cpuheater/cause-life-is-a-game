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
import ray
from gym.envs.classic_control import CartPoleEnv
from gym import spaces
import numpy as np
from copy import deepcopy
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.distributions import kl_divergence
from torch.nn.utils.rnn import pad_sequence

class CartPoleNoVelEnv(CartPoleEnv):
    """Variant of CartPoleEnv with velocity information removed. This task requires memory to solve."""

    def __init__(self):
        super(CartPoleNoVelEnv, self).__init__()
        high = np.array([
            self.x_threshold * 2,
            self.theta_threshold_radians * 2,
            ])
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

    @staticmethod
    def _pos_obs(full_obs):
        xpos, _xvel, thetapos, _thetavel = full_obs
        return xpos, thetapos

    def reset(self):
        full_obs = super().reset()
        return CartPoleNoVelEnv._pos_obs(full_obs)

    def step(self, action):
        full_obs, rew, done, info = super().step(action)
        return CartPoleNoVelEnv._pos_obs(full_obs), rew, done, info

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
    parser.add_argument('--num-envs', type=int, default=4,
                        help='the number of parallel game environment')
    parser.add_argument('--num-steps', type=int, default=128,
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
    parser.add_argument('--clip-vloss', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
                        help='Toggles wheter or not to use a clipped loss for the value function, as per the paper.')

    args = parser.parse_args()
    if not args.seed:
        args.seed = int(time.time())

args.batch_size = int(args.num_envs * args.num_steps)
args.minibatch_size = int(args.batch_size // args.n_minibatch)

class Buffer:
    def __init__(self, discount=0.99):
        self.discount = discount
        self.obs     = []
        self.actions    = []
        self.rewards    = []
        self.values     = []
        self.returns    = []
        self.advantages = []
        self.dones = []

        self.size = 0

        self.traj_idx = [0]
        self.buffer_ready = False

    def __len__(self):
        return len(self.obs)

    def push(self, obs, action, reward, value, done=False):
        self.obs  += [obs]
        self.actions += [action]
        self.rewards += [reward]
        self.values  += [value]

        self.size += 1

    def end_trajectory(self, terminal_value=0):
        self.traj_idx += [self.size]
        rewards = self.rewards[self.traj_idx[-2]:self.traj_idx[-1]]

        returns = []

        R = terminal_value
        for reward in reversed(rewards):
            R = self.discount * R + reward
            returns.insert(0, R)

        self.returns += returns

    def _finish_buffer(self):
        self.obs  = torch.Tensor(self.obs).to(device)
        self.actions = torch.Tensor(self.actions).to(device)
        self.rewards = torch.Tensor(self.rewards).to(device)
        self.returns = torch.Tensor(self.returns).to(device)
        self.values  = torch.Tensor(self.values).to(device)

        a = self.returns - self.values
        a = (a - a.mean()) / (a.std() + 1e-4)
        self.advantages = a
        self.buffer_ready = True

    def sample(self, batch_size=64):
        if not self.buffer_ready:
            self._finish_buffer()
        random_indices = SubsetRandomSampler(range(len(self.traj_idx)-1))
        sampler = BatchSampler(random_indices, batch_size, drop_last=True)

        for traj_indices in sampler:
          obs     = [self.obs[self.traj_idx[i]:self.traj_idx[i+1]]     for i in traj_indices]
          actions    = [self.actions[self.traj_idx[i]:self.traj_idx[i+1]]    for i in traj_indices]
          returns    = [self.returns[self.traj_idx[i]:self.traj_idx[i+1]]    for i in traj_indices]
          advantages = [self.advantages[self.traj_idx[i]:self.traj_idx[i+1]] for i in traj_indices]

          traj_mask  = [torch.ones_like(r) for r in returns]

          states     = pad_sequence(obs, batch_first=False)
          actions    = pad_sequence(actions, batch_first=False)
          returns    = pad_sequence(returns, batch_first=False)
          advantages = pad_sequence(advantages, batch_first=False)
          traj_mask  = pad_sequence(traj_mask, batch_first=False)

          yield states, actions, returns, advantages, traj_mask

def merge_buffers(buffers):
    memory = Buffer()

    for b in buffers:
        offset = len(memory)

        memory.obs  += b.obs
        memory.actions += b.actions
        memory.rewards += b.rewards
        memory.values  += b.values
        memory.returns += b.returns

        memory.traj_idx += [offset + i for i in b.traj_idx[1:]]
        memory.size     += b.size
    return memory

@ray.remote
class PPO_Worker:
    """
    A class representing a parallel worker used to explore the
    environment.
    """
    def __init__(self, agent, env_fn, gamma):
        torch.set_num_threads(1)
        self.gamma = gamma
        self.agent = deepcopy(agent)
        self.env = env_fn

    def sync_policy(self, new_agent_params):
        for p, new_p in zip(self.agent.parameters(), new_agent_params):
            p.data.copy_(new_p)

    def collect_experience(self, max_traj_len, min_steps):
        with torch.no_grad():
            num_steps = 0
            memory = Buffer(self.gamma)
            agent = self.agent

            while num_steps < min_steps:
                state = torch.Tensor(self.env.reset()).to(device)

                done = False
                value = 0
                traj_len = 0

                if hasattr(agent, 'init_hidden_state'):
                    agent.init_hidden_state()

                while not done and traj_len < max_traj_len:
                    action, log_probs, entropy = agent.get_action(state)
                    value = agent.get_value(state)
                    next_state, reward, done, info = self.env.step(action.cpu().numpy())

                    reward = np.array([reward])

                    memory.push(state.cpu().numpy(), np.array([action.cpu().numpy().item()]), reward, value.cpu().numpy())

                    state = torch.Tensor(next_state).to(device)

                    traj_len += 1
                    num_steps += 1
                    if 'episode' in info.keys():
                        print(f" episode_reward={info['episode']['r']}")
                        break

                value = (not done) * agent.get_value(state).cpu().numpy()
                memory.end_trajectory(terminal_value=value)

            return memory

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
        env = gym.make(gym_id) #CartPoleNoVelEnv()
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env
    return thunk


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


def create_layers(layer_fn, input_dim, layer_sizes):
    ret = nn.ModuleList()
    ret += [layer_fn(input_dim, layer_sizes[0])]
    for i in range(len(layer_sizes)-1):
        ret += [layer_fn(layer_sizes[i], layer_sizes[i+1])]
    return ret

class Agent(nn.Module):
    def __init__(self, in_dim, layers):
        super(Agent, self).__init__()
        self.layers = create_layers(nn.LSTMCell, in_dim, layers)
        self.actor = layer_init(nn.Linear(128, 2), std=0.01)
        self.critic = layer_init(nn.Linear(128, 1), std=1)
        self.init_hidden_state()
        self.calculate_norm = False


    def init_hidden_state(self, batch_size=1):
        self.hidden = [torch.zeros(batch_size, l.hidden_size).to(device) for l in self.layers]
        self.cells  = [torch.zeros(batch_size, l.hidden_size).to(device) for l in self.layers]

    def _forward(self, x):
        dims = len(x.size())
        if dims == 3: # if we get a batch of trajectories
            self.init_hidden_state(batch_size=x.size(1))

            if self.calculate_norm:
                self.latent_norm = 0

            y = []
            for t, x_t in enumerate(x):
                for idx, layer in enumerate(self.layers):
                    c, h = self.cells[idx], self.hidden[idx]
                    self.hidden[idx], self.cells[idx] = layer(x_t, (h, c))
                    x_t = self.hidden[idx]

                    if self.calculate_norm:
                        self.latent_norm += (torch.mean(torch.abs(x_t)) + torch.mean(torch.abs(self.cells[idx])))

                y.append(x_t)
            x = torch.stack([x_t for x_t in y])

            if self.calculate_norm:
                self.latent_norm /= len(x) * len(self.layers)
        else:
            if dims == 1: # if we get a single timestep (if not, assume we got a batch of single timesteps)
                x = x.view(1, -1)

            for idx, layer in enumerate(self.layers):
                h, c = self.hidden[idx], self.cells[idx]
                self.hidden[idx], self.cells[idx] = layer(x, (h, c))
                x = self.hidden[idx]

            if dims == 1:
                x = x.view(-1)
        return x

    def get_action(self, x, action=None):
        x = self._forward(x)
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy()

    def get_value(self, x):
        x = self._forward(x)
        return self.critic(x)

#envs = VecPyTorch(DummyVecEnv([make_env(args.gym_id, args.seed+i, i) for i in range(args.num_envs)]), device)

agent = Agent(4, layers=(128, 128)).to(device)
old_agent  = deepcopy(agent)

ray.init(local_mode=True)   #num_cpus=args.num_envs)
workers = [PPO_Worker.remote(agent, gym.wrappers.RecordEpisodeStatistics(gym.make(args.gym_id)), args.gamma) for _ in range(args.num_envs)]

#assert isinstance(envs.action_space, Discrete), "only discrete action space is supported"
optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

if args.anneal_lr:
    # https://github.com/openai/baselines/blob/ea25b9e8b234e6ee1bca43083f8f3cf974143998/baselines/ppo2/defaults.py#L20
    lr = lambda f: f * args.learning_rate

# ALGO Logic: Storage for epoch data
#obs = torch.zeros((args.num_steps, args.num_envs) + envs.observation_space.shape).to(device)
#actions = torch.zeros((args.num_steps, args.num_envs) + envs.action_space.shape).to(device)
#logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
#rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
#dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
#values = torch.zeros((args.num_steps, args.num_envs)).to(device)

# TRY NOT TO MODIFY: start the game
global_step = 0
# Note how `next_obs` and `next_done` are used; their usage is equivalent to
# https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/84a7582477fb0d5c82ad6d850fe476829dddd2e1/a2c_ppo_acktr/storage.py#L60
#next_obs = envs.reset()
next_done = torch.zeros(args.num_envs).to(device)
num_updates = args.total_timesteps // args.batch_size
for update in range(1, num_updates+1):
    global_step += 1 * args.num_envs
    # Annealing the rate if instructed to do so.
    if args.anneal_lr:
        frac = 1.0 - (update - 1.0) / num_updates
        lrnow = lr(frac)
        optimizer.param_groups[0]['lr'] = lrnow

    old_agent.load_state_dict(agent.state_dict())

    agent_param_id = ray.put(list(agent.parameters()))

    for w in workers:
        w.sync_policy.remote(agent_param_id)

    buffers = ray.get([w.collect_experience.remote(300, 1250) for w in workers])
    memory = merge_buffers(buffers)

    total_steps = len(memory)

    for i_epoch_pi in range(args.update_epochs):
        # for start in range(0, args.batch_size, args.minibatch_size):
        for batch in memory.sample(batch_size=64):
            states, actions, returns, advantages, mask = batch
            actions = actions.squeeze()
            with torch.no_grad():
                action, log_probs, entropy = old_agent.get_action(states, actions)
                log_probs = log_probs.sum(-1, keepdim=True)

            if args.norm_adv:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            action, new_log_probs, entropy = agent.get_action(states, actions)
            new_log_probs  = new_log_probs.sum(-1, keepdim=True)

            ratio      = ((new_log_probs - log_probs)).exp()
            cpi_loss   = ratio * advantages.squeeze() * mask.squeeze()
            clip_loss  = ratio.clamp(0.8, 1.2) * advantages.squeeze() * mask.squeeze()
            #clip_loss  = ratio.clamp(0.8, 1.2) * advantages.squeeze() * mask.squeeze()
            pg_loss = -torch.min(cpi_loss, clip_loss).mean()
            new_values = agent.get_value(states)
            v_loss = 0.5 * ((returns - new_values) * mask).pow(2).mean()
            loss = pg_loss + 0.5*v_loss + entropy.mean() * args.ent_coef
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=args.max_grad_norm)
            optimizer.step()
            approx_kl = (log_probs - new_log_probs).mean()

        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy.mean().item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)


    #    end = start + args.minibatch_size
    #    minibatch_ind = inds[start:end]
    #    mb_advantages = b_advantages[minibatch_ind]
    #    if args.norm_adv:
    #        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)



    #_, newlogproba, entropy = agent.get_action(b_obs[minibatch_ind], b_actions.long()[minibatch_ind])
    #ratio = (newlogproba - b_logprobs[minibatch_ind]).exp()

    # Stats
    #approx_kl = (b_logprobs[minibatch_ind] - newlogproba).mean()

    # Policy loss
    #pg_loss1 = -mb_advantages * ratio
    #pg_loss2 = -mb_advantages * torch.clamp(ratio, 1-args.clip_coef, 1+args.clip_coef)
    #pg_loss = torch.max(pg_loss1, pg_loss2).mean()
    #entropy_loss = entropy.mean()

    # Value loss
    #new_values = agent.get_value(b_obs[minibatch_ind]).view(-1)
    #if args.clip_vloss:
    #    v_loss_unclipped = ((new_values - b_returns[minibatch_ind]) ** 2)
    #    v_clipped = b_values[minibatch_ind] + torch.clamp(new_values - b_values[minibatch_ind], -args.clip_coef, args.clip_coef)
    #    v_loss_clipped = (v_clipped - b_returns[minibatch_ind])**2
    #    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
    #    v_loss = 0.5 * v_loss_max.mean()
    #else:
    #    v_loss = 0.5 * ((new_values - b_returns[minibatch_ind]) ** 2).mean()

    #loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

    #optimizer.zero_grad()
    #loss.backward()
    #nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
    #optimizer.step()

    #if args.kle_stop:
    #    if approx_kl > args.target_kl:
    #        break
    #if args.kle_rollback:
    #    if (b_logprobs[minibatch_ind] - agent.get_action(b_obs[minibatch_ind], b_actions.long()[minibatch_ind])[1]).mean() > args.target_kl:
    #        agent.load_state_dict(target_agent.state_dict())
    #        break

    # TRY NOT TO MODIFY: record rewards for plotting purposes
    #writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]['lr'], global_step)

    #if args.kle_stop or args.kle_rollback:
    #    writer.add_scalar("debug/pg_stop_iter", i_epoch_pi, global_step)

#envs.close()
writer.close()
