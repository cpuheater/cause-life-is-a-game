# https://github.com/facebookresearch/torchbeast/blob/master/torchbeast/core/environment.py

import numpy as np
from collections import deque
import gymnasium
from gymnasium import spaces
from gymnasium.spaces import Discrete, Box
import cv2
from vizdoom import GameVariable, Button
import skimage.transform
import gymnasium

cv2.ocl.setUseOpenCL(False)

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
import vizdoom
import argparse
from distutils.util import strtobool
import numpy as np
import time
import random
import os
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnvWrapper
from typing import Callable, Tuple, Dict, Optional
from stable_baselines3.common import vec_env
import itertools

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PPO agent')
    # Common arguments
    parser.add_argument('--exp-name', type=str, default=os.path.basename(__file__).rstrip(".py"),
                        help='the name of this experiment')
    parser.add_argument('--env-id', type=str, default="monsters",
                        help='the id of the gym environment')
    parser.add_argument('--learning-rate', type=float, default=4.5e-4,
                        help='the learning rate of the optimizer')
    parser.add_argument('--seed', type=int, default=1,
                        help='seed of the experiment')
    parser.add_argument('--total-timesteps', type=int, default=2000000,
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
    parser.add_argument('--frame-skip', type=int, default=2,
                        help='frame skip')
    parser.add_argument('--model-dir', type=str, default="models",
                        help='')

    # Algorithm specific arguments
    parser.add_argument('--n-minibatch', type=int, default=64,
                        help='the number of mini batch')
    parser.add_argument('--num-envs', type=int, default=64,
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
    parser.add_argument('--clip-coef', type=float, default=0.15,
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
    parser.add_argument('--channels', type=int, default=3,
                        help="the number of channels")

    args = parser.parse_args()
    #if not args.seed:
    args.seed = int(time.time())

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

class ViZDoomEnv(gymnasium.Env):

    def __init__(self,
                 game: vizdoom.DoomGame, channels = 1):
        super(ViZDoomEnv, self).__init__()
        h, w = game.get_screen_height(), game.get_screen_width()
        IMAGE_WIDTH, IMAGE_HEIGHT = 112, 64
        self.observation_space = spaces.Box(low=0, high=255, shape=(channels, IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.uint8)

        # Assign other variables
        self.game = game
        self.actions = self._get_actions()
        self.action_space = spaces.Discrete(len(self.actions))
        self.frame_skip = args.frame_skip
        self.scale_reward = args.scale_reward
        self.empty_frame = np.zeros(self.observation_space.shape, dtype=np.uint8)
        self.state = self.empty_frame
        self.channels = channels
        self.prev_pos = None

    def get_distance(self):
        curr_position = [self.game.get_game_variable(GameVariable.POSITION_X), self.game.get_game_variable(GameVariable.POSITION_Y)]
        dist = np.linalg.norm(np.array(curr_position) - np.array(self.prev_pos))
        self.prev_pos = curr_position
        return dist * 0.1

    def get_kill_reward(self):
        curr_killcount = self.game.get_game_variable(GameVariable.KILLCOUNT)
        killcount = curr_killcount - self.prev_killcount
        self.prev_killcount = curr_killcount
        return killcount * 100

    def get_ammo_reward(self):
        curr_ammo = self.game.get_game_variable(GameVariable.AMMO2)
        ammocount = curr_ammo - self.prev_ammo
        self.prev_ammo = curr_ammo
        return ammocount * 1

    def step(self, action: int):
        info = {}
        reward = self.game.make_action(self.actions[action], self.frame_skip)
        done = self.game.is_episode_finished()
        self.state = self._get_frame(done)
        #print(f"reward: {reward}, kill_reward: {self.get_kill_reward()}, distance: {self.get_distance()}, killcount: {self.game.get_game_variable(GameVariable.KILLCOUNT)}")
        reward += self.get_kill_reward() + self.get_distance() + self.get_ammo_reward()
        reward = reward * self.scale_reward
        #print(f"reward: {reward}")
        self.total_reward += reward
        self.total_length += 1
        if done:
            info['reward'] = self.total_reward
            info['length'] = self.total_length
            info['killcount'] = self.game.get_game_variable(GameVariable.KILLCOUNT)
        return self.state, reward, done, done, info

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self.game.set_seed(seed)
        self.game.new_episode()
        self.state = self._get_frame()
        self.prev_killcount = self.game.get_game_variable(GameVariable.KILLCOUNT)
        self.prev_pos = [self.game.get_game_variable(GameVariable.POSITION_X), self.game.get_game_variable(GameVariable.POSITION_Y)]
        self.prev_ammo = self.game.get_game_variable(GameVariable.AMMO2)
        self.total_length = 0
        self.total_reward = 0
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

    def _get_actions(self):
        possible_actions = create_actions(np.array([
            Button.TURN_LEFT, Button.TURN_RIGHT, Button.MOVE_FORWARD, Button.MOVE_BACKWARD, Button.ATTACK]))
        return possible_actions

def create_actions(actions):
    MUTUALLY_EXCLUSIVE_GROUPS = [
        #[Button.MOVE_RIGHT, Button.MOVE_LEFT],
        [Button.TURN_RIGHT, Button.TURN_LEFT],
        [Button.MOVE_FORWARD, Button.MOVE_BACKWARD],
    ]
    EXCLUSIVE_BUTTONS = [Button.ATTACK]
    def has_exclusive_button(actions: np.ndarray, buttons: np.array) -> np.array:
        exclusion_mask = np.isin(buttons, EXCLUSIVE_BUTTONS)
        return (np.any(actions.astype(bool) & exclusion_mask, axis=-1)) & (np.sum(actions, axis=-1) > 1)

    def has_excluded_pair(actions: np.ndarray, buttons: np.array) -> np.array:
        mutual_exclusion_mask = np.array([np.isin(buttons, excluded_group)
                                    for excluded_group in MUTUALLY_EXCLUSIVE_GROUPS])
        return np.any(np.sum(
            (actions[:, np.newaxis, :] * mutual_exclusion_mask.astype(int)),
            axis=-1) > 1, axis=-1)


    def get_available_actions(buttons: np.array):
        action_combinations = np.array([list(seq) for seq in itertools.product([0., 1.], repeat=len(buttons))])

        illegal_mask = (has_excluded_pair(action_combinations, buttons)
                | has_exclusive_button(action_combinations, buttons))

        possible_actions = action_combinations[~illegal_mask]
        possible_actions = possible_actions[np.sum(possible_actions, axis=1) > 0]
        return possible_actions.tolist()

    possible_actions = get_available_actions(actions)
    return possible_actions

class VecPyTorch(VecEnvWrapper):
    def __init__(self, venv, device):
        super(VecPyTorch, self).__init__(venv)
        self.device = device

    def reset(self):
        obs  = self.venv.reset()
        obs = torch.from_numpy(obs).float().to(self.device)
        return obs

    def step_async(self, actions):
        actions = actions.cpu().numpy()
        self.venv.step_async(actions)

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        #done = np.logical_or(terminations, truncations)
        obs = torch.from_numpy(obs).float().to(self.device)
        reward = torch.from_numpy(reward).unsqueeze(dim=1).float()
        return obs, reward, done, info


# taken from https://github.com/AIcrowd/neurips2020-procgen-starter-kit/blob/142d09586d2272a17f44481a115c4bd817cf6a94/models/impala_cnn_torch.py
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv0 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1)

    def forward(self, x):
        inputs = x
        x = nn.functional.relu(x)
        x = self.conv0(x)
        x = nn.functional.relu(x)
        x = self.conv1(x)
        return x + inputs


class ConvSequence(nn.Module):
    def __init__(self, input_shape, out_channels):
        super().__init__()
        self._input_shape = input_shape
        self._out_channels = out_channels
        self.conv = nn.Conv2d(in_channels=self._input_shape[0], out_channels=self._out_channels, kernel_size=3, padding=1)
        self.res_block0 = ResidualBlock(self._out_channels)
        self.res_block1 = ResidualBlock(self._out_channels)

    def forward(self, x):

        x = self.conv(x)
        x = nn.functional.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x = self.res_block0(x)
        x = self.res_block1(x)
        assert x.shape[1:] == self.get_output_shape()
        return x

    def get_output_shape(self):
        _c, h, w = self._input_shape
        return (self._out_channels, (h + 1) // 2, (w + 1) // 2)


class Agent(nn.Module):
        def __init__(self, num_actions, frames=1):
            super(Agent, self).__init__()

            c, h, w = envs.observation_space.shape
            shape = (c, h, w)
            conv_seqs = []
            for out_channels in [16, 32, 32]:
                conv_seq = ConvSequence(shape, out_channels)
                shape = conv_seq.get_output_shape()
                conv_seqs.append(conv_seq)
            conv_seqs += [
                nn.Flatten(),
                nn.ReLU(),
                nn.Linear(in_features=shape[0] * shape[1] * shape[2], out_features=256),
                nn.ReLU()
            ]
            self.network = nn.Sequential(*conv_seqs)

            self.actor = layer_init(nn.Linear(256, num_actions), std=0.01)
            self.critic = layer_init(nn.Linear(256, 1), std=1)

        def forward(self, x):
            x = x / 255
            x = self.network(x)
            return x

        def get_action(self, x, action=None):
            x = self.forward(x)
            value = self.critic(x)
            logits = self.actor(x)
            probs = Categorical(logits=logits)
            if action is None:
                action = probs.sample()
            return value, action, probs.log_prob(action), probs.entropy()

        def get_value(self, x):
            return self.critic(self.forward(x))

if __name__ == "__main__":
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.n_minibatch)
    # TRY NOT TO MODIFY: setup the environment
    experiment_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
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

    def create_env() -> ViZDoomEnv:
        def thunk():
            game = vizdoom.DoomGame()
            game.load_config(f'scenarios/{args.env_id}.cfg')
            game.set_window_visible(False)
            game.init()
            # Wrap the game with the Gym adapter.
            return ViZDoomEnv(game, channels=args.channels)
        return thunk

    #envs = VecPyTorch(DummyVecEnv([create_env(**kwargs) for i in range(args.num_envs)]), device)

    envs = VecPyTorch(
            SubprocVecEnv([create_env() for i in range(args.num_envs)], "fork"),
            device
        )
    assert isinstance(envs.action_space, gymnasium.spaces.discrete.Discrete), "only discrete action space is supported"

    # ALGO LOGIC: initialize agent here:
    agent = Agent(envs.action_space.n, args.channels).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)#, weight_decay=0.000001)
    if args.anneal_lr:
        # https://github.com/openai/baselines/blob/ea25b9e8b234e6ee1bca43083f8f3cf974143998/baselines/ppo2/defaults.py#L20
        lr = lambda f: f * args.learning_rate

    # ALGO Logic: Storage for epoch data
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    # Note how `next_obs` and `next_done` are used; their usage is equivalent to
    # https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/84a7582477fb0d5c82ad6d850fe476829dddd2e1/a2c_ppo_acktr/storage.py#L60
    next_obs = envs.reset()
    next_done = torch.zeros(args.num_envs).to(device)
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

            # ALGO LOGIC: put action logic here
            with torch.no_grad():
                value, action, logproba, _ = agent.get_action(obs[step])

            values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logproba

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, rs, ds, infos = envs.step(action)
            rewards[step], next_done = rs.view(-1), torch.Tensor(ds).to(device)

            #for info in infos:
            #    if 'episode' in info.keys():
            #        print(f"global_step={global_step}, episode_reward={info['episode']['r']}")
            #        writer.add_scalar("charts/episode_reward", info['episode']['r'], global_step)
            #        break
            for info in infos:
                if 'reward' in info.keys():
                    print(f"global_step={global_step}, episodic_return={info['reward']}, episodic_killcount={info['killcount']}")
                    writer.add_scalar("charts/episodic_return", info['reward'], global_step)
                if 'length' in info.keys():
                    writer.add_scalar("charts/episodic_length", info['length'], global_step)
                if 'killcount' in info.keys():
                    writer.add_scalar("charts/episodic_killcount", info['killcount'], global_step)



        # bootstrap reward if not done. reached the batch limit
        with torch.no_grad():
            last_value = agent.get_value(next_obs.to(device)).reshape(1, -1)
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

        # flatten the batch
        b_obs = obs.reshape((-1,)+envs.observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,)+envs.action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizaing the policy and value network
        inds = np.arange(args.batch_size,)
        for i_epoch_pi in range(args.update_epochs):
            np.random.shuffle(inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                minibatch_ind = inds[start:end]
                mb_advantages = b_advantages[minibatch_ind]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                _, _, newlogproba, entropy = agent.get_action(b_obs[minibatch_ind], b_actions.long()[minibatch_ind])
                ratio = (newlogproba - b_logprobs[minibatch_ind]).exp()

                # Stats
                approx_kl = (b_logprobs[minibatch_ind] - newlogproba).mean()

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1-args.clip_coef, 1+args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                entropy_loss = entropy.mean()

                # Value loss
                new_values = agent.get_value(b_obs[minibatch_ind]).view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = ((new_values - b_returns[minibatch_ind]) ** 2)
                    v_clipped = b_values[minibatch_ind] + torch.clamp(new_values - b_values[minibatch_ind], -args.clip_coef, args.clip_coef)
                    v_loss_clipped = (v_clipped - b_returns[minibatch_ind])**2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((new_values - b_returns[minibatch_ind]) ** 2).mean()

                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.kle_stop:
                if approx_kl > args.target_kl:
                    break

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]['lr'], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy.mean().item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        if args.kle_stop or args.kle_rollback:
            writer.add_scalar("debug/pg_stop_iter", i_epoch_pi, global_step)
        writer.add_scalar("charts/sps", int(global_step / (time.time() - start_time)), global_step)

    torch.save(agent, f"agent_{args.total_timesteps}_{time.time()}.pt")

    envs.close()
    writer.close()