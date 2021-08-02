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

class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, noop_max=30):
        """Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.
        """
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def reset(self, **kwargs):
        """ Do no-op action for a number of steps in [1, noop_max]."""
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.randint(1, self.noop_max + 1) #pylint: disable=E1101
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)

class FireResetEnv(gym.Wrapper):
    def __init__(self, env):
        """Take action on reset for environments that are fixed until firing."""
        gym.Wrapper.__init__(self, env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)

class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env):
        """Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.
        """
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done  = True

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            # for Qbert sometimes we stay in lives == 0 condition for a few frames
            # so it's important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
        self.lives = lives
        return obs, reward, done, info

    def reset(self, **kwargs):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            obs = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs

class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        """Return only every `skip`-th frame"""
        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = np.zeros((2,)+env.observation_space.shape, dtype=np.uint8)
        self._skip       = skip

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            if i == self._skip - 2: self._obs_buffer[0] = obs
            if i == self._skip - 1: self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
        # Note that the observation on the done=True frame
        # doesn't matter
        max_frame = self._obs_buffer.max(axis=0)

        return max_frame, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

class ClipRewardEnv(gym.RewardWrapper):
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)

    def reward(self, reward):
        """Bin reward to {+1, 0, -1} by its sign."""
        return np.sign(reward)


class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env, width=84, height=84, grayscale=True, dict_space_key=None):
        """
        Warp frames to 84x84 as done in the Nature paper and later work.
        If the environment uses dictionary observations, `dict_space_key` can be specified which indicates which
        observation should be warped.
        """
        super().__init__(env)
        self._width = width
        self._height = height
        self._grayscale = grayscale
        self._key = dict_space_key
        if self._grayscale:
            num_colors = 1
        else:
            num_colors = 3

        new_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self._height, self._width, num_colors),
            dtype=np.uint8,
        )
        if self._key is None:
            original_space = self.observation_space
            self.observation_space = new_space
        else:
            original_space = self.observation_space.spaces[self._key]
            self.observation_space.spaces[self._key] = new_space
        assert original_space.dtype == np.uint8 and len(original_space.shape) == 3

    def observation(self, obs):
        if self._key is None:
            frame = obs
        else:
            frame = obs[self._key]

        if self._grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(
            frame, (self._width, self._height), interpolation=cv2.INTER_AREA
        )
        if self._grayscale:
            frame = np.expand_dims(frame, -1)

        if self._key is None:
            obs = frame
        else:
            obs = obs.copy()
            obs[self._key] = frame
        return obs


class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        """Stack k last frames.
        Returns lazy array, which is much more memory efficient.
        See Also
        --------
        baselines.common.atari_wrappers.LazyFrames
        """
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = spaces.Box(low=0, high=255, shape=((shp[0] * k,)+shp[1:]), dtype=env.observation_space.dtype)

    def reset(self):
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames))

class ScaledFloatFrame(gym.ObservationWrapper):
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=env.observation_space.shape, dtype=np.float32)

    def observation(self, observation):
        # careful! This undoes the memory optimization, use
        # with smaller replay buffers only.
        return np.array(observation).astype(np.float32) / 255.0

class LazyFrames(object):
    def __init__(self, frames):
        """This object ensures that common frames between the observations are only stored once.
        It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
        buffers.
        This object should only be converted to numpy array before being passed to the model.
        You'd not believe how complex the previous solution was."""
        self._frames = frames
        self._out = None

    def _force(self):
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=0)
            self._frames = None
        return self._out

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[i]

    def count(self):
        frames = self._force()
        return frames.shape[frames.ndim - 1]

    def frame(self, i):
        return self._force()[..., i]

def wrap_atari(env, max_episode_steps=None):
    assert 'NoFrameskip' in env.spec.id
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)

    assert max_episode_steps is None

    return env

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

def wrap_deepmind(env, episode_life=False, clip_rewards=True, frame_stack=False, scale=False):
    """Configure environment for DeepMind-style Atari.
    """
    if episode_life:
        env = EpisodicLifeEnv(env)
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = WarpFrame(env)
    if scale:
        env = ScaledFloatFrame(env)
    if clip_rewards:
        env = ClipRewardEnv(env)
    env = ImageToPyTorch(env)
    if frame_stack:
        env = FrameStack(env, 4)
    return env



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SAC with 2 Q functions, Online updates')
    # Common arguments
    parser.add_argument('--exp-name', type=str, default=os.path.basename(__file__).rstrip(".py"),
                        help='the name of this experiment')
    parser.add_argument('--gym-id', type=str, default="MsPacmanNoFrameskip-v4",
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
    parser.add_argument('--buffer-size', type=int, default=135000,
                        help='the replay memory buffer size')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='the discount factor gamma')
    parser.add_argument('--target-network-frequency', type=int, default=8000, # Denis Yarats' implementation delays this by 2.
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
    parser.add_argument('--policy-lr', type=float, default=3e-4,
                        help='the learning rate of the policy network optimizer')
    parser.add_argument('--q-lr', type=float, default=3e-4,
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
env = gym.make(args.gym_id)
env = wrap_atari(env)
env = wrap_deepmind(
    env,
    clip_rewards=True,
    frame_stack=True,
    scale=False,
)
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = args.torch_deterministic
env.seed(args.seed)
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

def layer_init(layer, weight_gain=1, bias_const=0):
    if isinstance(layer, nn.Linear):
        if args.weights_init == "xavier":
            torch.nn.init.xavier_uniform_(layer.weight, gain=weight_gain)
        elif args.weights_init == "orthogonal":
            torch.nn.init.orthogonal_(layer.weight, gain=weight_gain)
        if args.bias_init == "zeros":
            torch.nn.init.constant_(layer.bias, bias_const)

def conv_shape(input, kernel_size, stride, padding=0):
    return (input + 2 * padding - kernel_size) // stride + 1

class Policy(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(Policy, self).__init__()
        #self.fc1 = nn.Linear(input_shape, 256) # Better result with slightly wider networks.
        #self.fc2 = nn.Linear(256, 128)
        #self.mean = nn.Linear(128, output_shape)
        #self.logstd = nn.Linear(128, output_shape)
        # action rescaling
        #self.action_scale = torch.FloatTensor(
        #    (env.action_space.high - env.action_space.low) / 2.)
        #self.action_bias = torch.FloatTensor(
        #    (env.action_space.high + env.action_space.low) / 2.)
        #self.apply(layer_init)

        self.input_shape = input_shape
        self.num_actions = num_actions

        c, w, h = self.input_shape

        self.conv1 = nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4, padding=0)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=0)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0)

        conv1_out_w = conv_shape(w, 8, 4)
        conv1_out_h = conv_shape(h, 8, 4)
        conv2_out_w = conv_shape(conv1_out_w, 4, 2)
        conv2_out_h = conv_shape(conv1_out_h, 4, 2)
        conv3_out_w = conv_shape(conv2_out_w, 3, 1)
        conv3_out_h = conv_shape(conv2_out_h, 3, 1)

        flatten_size = conv3_out_w * conv3_out_h * 64

        self.fc = nn.Linear(in_features=flatten_size, out_features=512)
        self.logits = nn.Linear(in_features=512, out_features=self.num_actions)

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
        x = x / 255.0
        #x = F.relu(self.fc1(x))
        #x = F.relu(self.fc2(x))
        #mean = self.mean(x)
        #log_std = self.logstd(x)
        #log_std = torch.tanh(log_std)
        #log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1) # From SpinUp / Denis Yarats

        #return mean, log_std
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
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
        #mean, log_std = self.forward(x, device)
        #std = log_std.exp()
        #normal = Normal(mean, std)
        #x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        #y_t = torch.tanh(x_t)
        #action = y_t * self.action_scale + self.action_bias
        #log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        #log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) +  1e-6)
        #log_prob = log_prob.sum(1, keepdim=True)
        #mean = torch.tanh(mean) * self.action_scale + self.action_bias
        #return action, log_prob, mean


        probs, log_probs = self.forward(x, device)
        dist = Categorical(probs)
        action = dist.sample()
        return action.detach().cpu().numpy()[0], dist

    #def to(self, device):
    #    self.action_scale = self.action_scale.to(device)
    #    self.action_bias = self.action_bias.to(device)
    #    return super(Policy, self).to(device)

class SoftQNetwork(nn.Module):
    def __init__(self, input_shape, num_actions, layer_init):
        super(SoftQNetwork, self).__init__()
        #self.fc1 = nn.Linear(input_shape+output_shape, 256)
        #self.fc2 = nn.Linear(256, 128)
        #self.fc3 = nn.Linear(128, 1)
        #self.apply(layer_init)
        self.state_shape = input_shape
        self.n_actions = num_actions

        c, w, h = self.state_shape

        self.conv1 = nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4, padding=0)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=0)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0)

        conv1_out_w = conv_shape(w, 8, 4)
        conv1_out_h = conv_shape(h, 8, 4)
        conv2_out_w = conv_shape(conv1_out_w, 4, 2)
        conv2_out_h = conv_shape(conv1_out_h, 4, 2)
        conv3_out_w = conv_shape(conv2_out_w, 3, 1)
        conv3_out_h = conv_shape(conv2_out_h, 3, 1)

        flatten_size = conv3_out_w * conv3_out_h * 64

        self.fc = nn.Linear(in_features=flatten_size, out_features=512)
        self.q_value = nn.Linear(in_features=512, out_features=self.n_actions)

        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
                layer.bias.data.zero_()

        nn.init.kaiming_normal_(self.fc.weight, nonlinearity="relu")
        self.fc.bias.data.zero_()
        nn.init.xavier_uniform_(self.q_value.weight)
        self.q_value.bias.data.zero_()

    def forward(self, x, device):
        #x = torch.Tensor(x).to(device)
        #x = torch.cat([x, a], 1)
        #x = F.relu(self.fc1(x))
        #x = F.relu(self.fc2(x))
        #x = self.fc3(x)
        #return x
        x = torch.Tensor(x).to(device)
        x = x / 255.0
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.contiguous()
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))
        return self.q_value(x)

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

rb = ReplayBuffer(args.buffer_size)
pg = Policy(input_shape, env.action_space.n).to(device)
qf1 = SoftQNetwork(input_shape, env.action_space.n, layer_init).to(device)
qf2 = SoftQNetwork(input_shape, env.action_space.n, layer_init).to(device)
qf1_target = SoftQNetwork(input_shape, env.action_space.n, layer_init).to(device)
qf2_target = SoftQNetwork(input_shape, env.action_space.n, layer_init).to(device)
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
    reward = np.sign(reward)
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
            print(f"target update {global_step}")
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
        max_episode_reward = max(max_episode_reward, episode_reward)
        writer.add_scalar("charts/max_episode_reward", max_episode_reward, global_step)
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
