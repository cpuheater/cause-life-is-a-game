import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
import argparse
from distutils.util import strtobool
import collections
import numpy as np
import gymnasium
from gymnasium import spaces
from gymnasium.spaces import Discrete, Box
import time
import random
import os
from vizdoom import DoomGame, Mode, ScreenFormat, ScreenResolution
import skimage.transform
import vizdoom
import cv2


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
        print(f"reward: {reward}")
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
        return screen

    def _get_frame(self, done: bool = False) -> np.ndarray:
        return self.get_screen() if not done else self.empty_frame


def create_env() -> ViZDoomEnv:
    game = vizdoom.DoomGame()
    game.load_config(f'scenarios/{args.env_id}.cfg')
    game.set_window_visible(True)
    game.init()
    return ViZDoomEnv(game, channels=args.channels)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SAC')
    # Common arguments
    parser.add_argument('--exp-name', type=str, default=os.path.basename(__file__).rstrip(".py"),
                        help='the name of this experiment')
    parser.add_argument('--env-id', type=str, default="basic",
                        help='the id of the gym environment')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
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
    parser.add_argument('--track', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
                        help='run the script in production mode and use wandb to log outputs')
    parser.add_argument('--capture-video', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
                        help='weather to capture videos of the agent performances (check out `videos` folder)')
    parser.add_argument('--wandb-project-name', type=str, default="cleanRL",
                        help="the wandb's project name")
    parser.add_argument('--wandb-entity', type=str, default=None,
                        help="the entity (team) of wandb's project")
    parser.add_argument('--autotune', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
                        help='automatic tuning of the entropy coefficient.')
    parser.add_argument('--scale-reward', type=float, default=0.01,
                        help='scale reward')
    parser.add_argument('--channels', type=int, default=1,
                        help="the number of channels")
    parser.add_argument('--frame-skip', type=int, default=4,
                        help='frame skip')

    # Algorithm specific arguments
    parser.add_argument('--buffer-size', type=int, default=135000,
                        help='the replay memory buffer size')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='the discount factor gamma')
    parser.add_argument('--target-network-frequency', type=int, default=2, # Denis Yarats' implementation delays this by 2.
                        help="the timesteps it takes to update the target network")
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
    parser.add_argument('--policy-lr', type=float, default=1e-4,
                        help='the learning rate of the policy network optimizer')
    parser.add_argument('--q-lr', type=float, default=1e-4,
                        help='the learning rate of the Q network network optimizer')
    parser.add_argument('--policy-frequency', type=int, default=1,
                        help='delays the update of the actor, as per the TD3 paper.')
    # NN Parameterization
    parser.add_argument('--weights-init', default='xavier', const='orthogonal', nargs='?', choices=['xavier', "orthogonal", 'uniform'],
                        help='weight initialization scheme for the neural networks.')
    parser.add_argument('--bias-init', default='zeros', const='xavier', nargs='?', choices=['zeros', 'uniform'],
                        help='weight initialization scheme for the neural networks.')

    args = parser.parse_args()
    #if not args.seed:
    args.seed = int(time.time())

# TRY NOT TO MODIFY: setup the environment
experiment_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
writer = SummaryWriter(f"runs/{experiment_name}")
writer.add_text('hyperparameters', "|param|value|\n|-|-|\n%s" % (
    '\n'.join([f"|{key}|{value}|" for key, value in vars(args).items()])))
if args.track:
    import wandb
    wandb.init(project=args.wandb_project_name, entity=args.wandb_entity, sync_tensorboard=True, config=vars(args), name=experiment_name, monitor_gym=True, save_code=True)
    writer = SummaryWriter(f"/tmp/{experiment_name}")


# TRY NOT TO MODIFY: seeding
device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')
env = create_env()
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = args.torch_deterministic
env.action_space.seed(args.seed)
env.observation_space.seed(args.seed)
# ALGO LOGIC: initialize agent here:

def layer_init(layer, weight_gain=np.sqrt(2), bias_const=0):
    if isinstance(layer, nn.Linear):
        if args.weights_init == "xavier":
            torch.nn.init.xavier_uniform_(layer.weight, gain=weight_gain)
        elif args.weights_init == "orthogonal":
            torch.nn.init.orthogonal_(layer.weight, gain=weight_gain)
        if args.bias_init == "zeros":
            torch.nn.init.constant_(layer.bias, bias_const)

class Policy(nn.Module):
    def __init__(self, num_actions):
        super(Policy, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(args.channels, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2560, 512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=num_actions)
        )
        self.apply(layer_init)

    def forward(self, x, device):
        x = torch.Tensor(x).to(device)
        x = x / 255.0
        logits = self.network(x)
        probs = F.softmax(logits, -1)
        z = probs == 0
        z = z.float() * 1e-8
        log_probs = torch.log(probs + z)
        return probs, log_probs

    def get_action(self, x, device):
        probs, log_probs = self.forward(x, device)
        dist = Categorical(probs)
        action = dist.sample()
        return action.detach().cpu().numpy()[0]


class SoftQNetwork(nn.Module):
    def __init__(self, num_actions):
        super(SoftQNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(args.channels, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2560, 512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=num_actions)
        )
        self.apply(layer_init)

    def forward(self, x, device):
        x = torch.Tensor(x).to(device)
        x = x / 255.0
        x = self.network(x)
        return x

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
pg = Policy(env.action_space.n).to(device)
qf1 = SoftQNetwork(env.action_space.n).to(device)
qf2 = SoftQNetwork(env.action_space.n).to(device)
qf1_target = SoftQNetwork(env.action_space.n).to(device)
qf2_target = SoftQNetwork(env.action_space.n).to(device)
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
obs, _= env.reset()

for global_step in range(1, args.total_timesteps+1):
    # ALGO LOGIC: put action logic here
    if global_step < args.learning_starts:
        action = env.action_space.sample()
    else:
        action = pg.get_action([obs], device)

    # TRY NOT TO MODIFY: execute the game and log data.
    next_obs, reward, done, infos = env.step(action)
    if 'reward' in infos.keys():
        print(f"global_step={global_step}, episodic_return={infos['reward']}")
        writer.add_scalar("charts/episodic_return", infos['reward'], global_step)
    if 'length' in infos.keys():
        writer.add_scalar("charts/episodic_length", infos['length'], global_step)

    rb.put((obs, action, reward, next_obs, done))
    obs = next_obs

    # ALGO LOGIC: training.
    if len(rb.buffer) > args.batch_size and global_step % 4 == 0: # starts update as soon as there is enough data.
        s_obs, s_actions, s_rewards, s_next_obses, s_dones = rb.sample(args.batch_size)
        with torch.no_grad():
            next_state_probs, next_state_log_probs = pg.forward(s_next_obses, device)
            qf1_next_target = qf1_target.forward(s_next_obses, device)
            qf2_next_target = qf2_target.forward(s_next_obses, device)
            min_qf_next_target = (next_state_probs * (torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_probs)).sum(dim=1)
            next_q_value = torch.Tensor(s_rewards).to(device) + (1 - torch.Tensor(s_dones).to(device)) * args.gamma * min_qf_next_target

        qf1_a_values = qf1.forward(s_obs, device)[np.arange(args.batch_size), np.array(s_actions)]
        qf2_a_values = qf2.forward(s_obs, device)[np.arange(args.batch_size), np.array(s_actions)]
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
                    alpha_loss = ( -log_alpha * (log_probs + target_entropy)).mean()

                    a_optimizer.zero_grad()
                    alpha_loss.backward()
                    a_optimizer.step()
                    alpha = log_alpha.exp().item()

        # update the target network
        if global_step % args.target_network_frequency == 0:
            #qf1_target.load_state_dict(qf1.state_dict())
            #qf1_target.eval()
            #qf2_target.load_state_dict(qf2.state_dict())
            #qf2_target.eval()
            for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
            for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

    if len(rb.buffer) > args.learning_starts and global_step % 100 == 0:
        writer.add_scalar("losses/soft_q_value_1_loss", qf1_loss.item(), global_step)
        writer.add_scalar("losses/soft_q_value_2_loss", qf2_loss.item(), global_step)
        writer.add_scalar("losses/qf_loss", qf_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", policy_loss.item(), global_step)
        writer.add_scalar("losses/alpha", alpha, global_step)
        if args.autotune:
            writer.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)

    if done:
        obs, _ = env.reset()    

writer.close()
env.close()
