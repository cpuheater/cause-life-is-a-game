# https://github.com/facebookresearch/torchbeast/blob/master/torchbeast/core/environment.py

import numpy as np
from collections import deque
import gym
from gym import spaces
import cv2
from vizdoom import DoomGame, Mode, ScreenFormat, ScreenResolution
import skimage.transform

cv2.ocl.setUseOpenCL(False)

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
import pybullet_envs
from gym.spaces import Discrete, Box, MultiBinary, MultiDiscrete, Space
import time
import random
import os
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnvWrapper

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PPO agent')
    # Common arguments
    parser.add_argument('--exp-name', type=str, default=os.path.basename(__file__).rstrip(".py"),
                        help='the name of this experiment')
    parser.add_argument('--gym-id', type=str, default="basic",
                        help='the id of the gym environment')
    parser.add_argument('--learning-rate', type=float, default=6e-4,
                        help='the learning rate of the optimizer')
    parser.add_argument('--seed', type=int, default=1,
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
    parser.add_argument('--scale-reward', type=float, default=0.01,
                        help='scale reward')
    parser.add_argument('--frame-skip', type=int, default=4,
                        help='frame skip')
    parser.add_argument('--rnn-hidden-size', type=int, default=256,
                        help='rnn hidden size')

    # Algorithm specific arguments
    parser.add_argument('--n-minibatch', type=int, default=8,
                        help='the number of mini batch')
    parser.add_argument('--num-envs', type=int, default=16,
                        help='the number of parallel game environment')
    parser.add_argument('--num-steps', type=int, default=5,
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
    parser.add_argument('--clip-coef', type=float, default=0.1,
                        help="the surrogate clipping coefficient")
    parser.add_argument('--update-epochs', type=int, default=4,
                         help="the K epochs to update the policy")
    parser.add_argument('--kle-stop', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
                         help='If toggled, the policy updates will be early stopped w.r.t target-kl')
    parser.add_argument('--kle-rollback', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
                         help='If toggled, the policy updates will roll back to previous policy if KL exceeds target-kl')
    parser.add_argument('--target-kl', type=float, default=0.03,
                         help='the target-kl variable that is referred by --kl')
    parser.add_argument('--gae', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
                         help='Use GAE for advantage computation')
    parser.add_argument('--norm-adv', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
                          help="Toggles advantages normalization")
    parser.add_argument('--anneal-lr', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
                          help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument('--clip-vloss', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
                          help='Toggles wheter or not to use a clipped loss for the value function, as per the paper.')

    args = parser.parse_args()
    #if not args.seed:
    args.seed = int(time.time())

args.batch_size = int(args.num_envs * args.num_steps)
args.minibatch_size = int(args.batch_size // args.n_minibatch)


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
def make_env(seed):
    def thunk():
        env = ViZDoomEnv(seed, args.gym_id, render=False, reward_scale=args.scale_reward, frame_skip=args.frame_skip)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env
    return thunk

#envs = VecPyTorch(DummyVecEnv([make_env(args.gym_id, args.seed+i, i) for i in range(args.num_envs)]), device)
# if args.prod_mode:
envs = VecPyTorch(
         SubprocVecEnv([make_env(args.seed+i) for i in range(args.num_envs)], "fork"),
         device
     )
assert isinstance(envs.action_space, Discrete), "only discrete action space is supported"

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



def recurrent_generator(obs, logprobs, actions, advantages, returns, values, rnn_hidden_states, masks):
        def _flatten_helper(T, N, _tensor):
            return _tensor.view(T * N, *_tensor.size()[2:])

        assert args.num_envs >= args.n_minibatch, (
        "PPO requires the number of envs ({}) "
        "to be greater than or equal to the number of "
        "PPO mini batches ({}).".format(args.num_envs, args.n_minibatch))
        num_envs_per_batch = args.num_envs // args.n_minibatch
        perm = torch.randperm(args.num_envs)
        for start_ind in range(0, args.num_envs, num_envs_per_batch):

            a_rnn_hidden_states = []
            a_obs = []
            a_actions = []
            a_values = []
            a_returns = []
            a_masks = []
            a_logprobs = []
            a_advantages = []

            for offset in range(num_envs_per_batch):
                ind = perm[start_ind + offset]
                a_rnn_hidden_states.append(rnn_hidden_states[0:1, ind])
                a_obs.append(obs[:, ind])
                a_actions.append(actions[:, ind])
                a_values.append(values[:, ind])
                a_returns.append(returns[:, ind])
                a_masks.append(masks[:, ind])
                a_logprobs.append(logprobs[:, ind])
                a_advantages.append(advantages[:, ind])


            T, N = args.num_steps, num_envs_per_batch

            b_rnn_hidden_states = torch.stack(a_rnn_hidden_states, 1).view(N, -1)
            b_obs = _flatten_helper(T, N, torch.stack(a_obs, 1))
            b_actions = _flatten_helper(T, N, torch.stack(a_actions, 1))
            b_values = _flatten_helper(T, N, torch.stack(a_values, 1))
            b_return = _flatten_helper(T, N, torch.stack(a_returns, 1))
            b_masks = _flatten_helper(T, N, torch.stack(a_masks, 1))
            b_logprobs = _flatten_helper(T, N, torch.stack(a_logprobs, 1))
            b_advantages = _flatten_helper(T, N, torch.stack(a_advantages, 1))

            yield b_obs, b_rnn_hidden_states, b_actions, \
                b_values, b_return, b_masks, b_logprobs, b_advantages



class Agent(nn.Module):
    def __init__(self, envs, frames=3, rnn_input_size=512, rnn_hidden_size=512):
        super(Agent, self).__init__()
        self.network = nn.Sequential(
            Scale(1/255),
            layer_init(nn.Conv2d(frames, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(2560, rnn_hidden_size)),
            nn.ReLU()
        )

        self.gru = nn.GRU(rnn_input_size, rnn_hidden_size)
        for name, param in self.gru.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                nn.init.orthogonal_(param, np.sqrt(2))

        self.actor = layer_init(nn.Linear(rnn_hidden_size, envs.action_space.n), std=0.01)
        self.critic = layer_init(nn.Linear(rnn_hidden_size, 1), std=1)

    def forward(self, x, hxs, mask):
        x = self.network(x)
        if x.size(0) == hxs.size(0):
            x, hxs = self.gru(x.unsqueeze(0), (hxs * mask).unsqueeze(0))
            x = x.squeeze()
            hxs = hxs.squeeze(0)
        else:
            N = hxs.size(0)
            T = int(x.size(0) / N)
            x = x.view(T, N, x.size(1))
            masks = mask.view(T, N)
            has_zeros = ((masks[1:] == 0.0) \
                         .any(dim=-1)
                         .nonzero()
                         .squeeze()
                         .cpu())

            if has_zeros.dim() == 0:
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()

            has_zeros = [0] + has_zeros + [T]
            hxs = hxs.unsqueeze(0)
            outputs = []
            for i in range(len(has_zeros) - 1):
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]

                rnn_scores, hxs = self.gru(
                    x[start_idx:end_idx],
                    hxs * masks[start_idx].view(1, -1, 1))

                outputs.append(rnn_scores)

            x = torch.cat(outputs, dim=0)
            x = x.view(T * N, -1)
            hxs = hxs.squeeze(0)
        return x, hxs

    def get_action(self, x, rnn_hidden_state, mask, action=None):
        x, rnn_hidden_state = self.forward(x, rnn_hidden_state, mask)
        value = self.critic(x)
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return value, rnn_hidden_state, action, probs.log_prob(action), probs.entropy()

    def get_value(self, x, rnn_hidden_state, mask):
        x, rnn_hidden_state = self.forward(x, rnn_hidden_state, mask)
        return self.critic(x)

agent = Agent(envs, rnn_hidden_size=args.rnn_hidden_size, rnn_input_size=args.rnn_hidden_size).to(device)
optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
if args.anneal_lr:
    # https://github.com/openai/baselines/blob/ea25b9e8b234e6ee1bca43083f8f3cf974143998/baselines/ppo2/defaults.py#L20
    lr = lambda f: f * args.learning_rate


rnn_hidden_size = args.rnn_hidden_size

# ALGO Logic: Storage for epoch data
obs = torch.zeros((args.num_steps, args.num_envs) + envs.observation_space.shape).to(device)
actions = torch.zeros((args.num_steps, args.num_envs) + envs.action_space.shape).to(device)
logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
values = torch.zeros((args.num_steps, args.num_envs)).to(device)
rnn_hidden_states = torch.zeros((args.num_steps, args.num_envs, rnn_hidden_size)).to(device)
masks = torch.ones((args.num_steps, args.num_envs, 1)).to(device)


# TRY NOT TO MODIFY: start the game
global_step = 0
# Note how `next_obs` and `next_done` are used; their usage is equivalent to
# https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/84a7582477fb0d5c82ad6d850fe476829dddd2e1/a2c_ppo_acktr/storage.py#L60
next_obs = envs.reset()
next_done = torch.zeros(args.num_envs).to(device)
mask = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in next_done])
rnn_hidden_state = torch.zeros((args.num_envs, rnn_hidden_size))
num_updates = args.total_timesteps // args.batch_size
for update in range(1, num_updates+1):
    # Annealing the rate if instructed to do so.
    if args.anneal_lr:
        frac = 1.0 - (update - 1.0) / num_updates
        lrnow = lr(frac)
        optimizer.param_groups[0]['lr'] = lrnow

    # TRY NOT TO MODIFY: prepare the execution of the game.
    for step in range(0, args.num_steps):
        global_step += 1 * args.num_envs
        obs[step] = next_obs
        dones[step] = next_done
        rnn_hidden_states[step] = rnn_hidden_state
        masks[step] = mask
        # ALGO LOGIC: put action logic here
        with torch.no_grad():
            value, rnn_hidden_state, action, logproba, _ = agent.get_action(obs[step], rnn_hidden_states[step], masks[step])

        actions[step] = action
        logprobs[step] = logproba
        values[step] = value.flatten()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rs, ds, infos = envs.step(action)
        rewards[step], next_done = rs.view(-1), torch.Tensor(ds).to(device)
        mask = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in next_done]).to(device)

        for info in infos:
            if 'Episode_Total_Reward' in info.keys():
                writer.add_scalar("charts/episode_reward", info['Episode_Total_Reward'], global_step)
            if 'Episode_Total_Len' in info.keys():
                writer.add_scalar("charts/episode_length", info['Episode_Total_Len'], global_step)

    # bootstrap reward if not done. reached the batch limit
    with torch.no_grad():
        last_value = agent.get_value(next_obs.to(device), rnn_hidden_state, mask).reshape(1, -1)
        if args.gae:
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = last_value
                else:
                    nextnonterminal = 1.0 - dones[t+1]
                    nextvalues = values[t+1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values
        else:
            returns = torch.zeros_like(rewards).to(device)
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    next_return = last_value
                else:
                    nextnonterminal = 1.0 - dones[t+1]
                    next_return = returns[t+1]
                returns[t] = rewards[t] + args.gamma * nextnonterminal * next_return
            advantages = returns - values


    # Optimizaing the policy and value network
    target_agent = Agent(envs, rnn_hidden_size=args.rnn_hidden_size, rnn_input_size=args.rnn_hidden_size).to(device)
    for i_epoch_pi in range(args.update_epochs):
        target_agent.load_state_dict(agent.state_dict())
        data_generator = recurrent_generator(obs, logprobs, actions, advantages, returns, values,
                                             rnn_hidden_states, masks)
        for batch in data_generator:
            b_obs, b_rnn_hidden_states, b_actions, b_values, b_returns, b_masks, b_logprobs, b_advantages = batch
            if args.norm_adv:
                b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

            newvalues, _, _, newlogproba, entropy = agent.get_action(b_obs, b_rnn_hidden_states, b_masks, b_actions.long())
            ratio = (newlogproba - b_logprobs).exp()

            # Stats
            approx_kl = (b_logprobs - newlogproba).mean()

            # Policy loss
            pg_loss1 = -b_advantages * ratio
            pg_loss2 = -b_advantages * torch.clamp(ratio, 1-args.clip_coef, 1+args.clip_coef)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()
            entropy_loss = entropy.mean()

            # Value loss
            new_values = newvalues.view(-1)
            if args.clip_vloss:
                v_loss_unclipped = ((new_values - b_returns) ** 2)
                v_clipped = b_values + torch.clamp(new_values - b_values, -args.clip_coef, args.clip_coef)
                v_loss_clipped = (v_clipped - b_returns)**2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()
            else:
                v_loss = 0.5 * ((new_values - b_returns) ** 2).mean()

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
    writer.add_histogram("actor/weights", agent.actor.weight)
    if args.kle_stop or args.kle_rollback:
        writer.add_scalar("debug/pg_stop_iter", i_epoch_pi, global_step)

envs.close()
writer.close()
