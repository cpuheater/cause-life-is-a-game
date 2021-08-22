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
import numpy as np
import gym
from gym.wrappers import TimeLimit, Monitor
from gym.spaces import Discrete, Box, MultiBinary, MultiDiscrete, Space
import time
import random
import os
from collections import deque
from distutils.util import strtobool
import numpy as np
import gym
import gym_microrts
from gym.wrappers import TimeLimit, Monitor
from gym_microrts.envs.vec_env import MicroRTSGridModeVecEnv
from gym_microrts import microrts_ai
from gym.spaces import Discrete, Box, MultiBinary, MultiDiscrete, Space
import time
import random
import os
from stable_baselines3.common.vec_env import VecEnvWrapper, VecVideoRecorder
from jpype.types import JArray, JInt

class VecMonitor(VecEnvWrapper):
    def __init__(self, venv):
        VecEnvWrapper.__init__(self, venv)
        self.eprets = None
        self.eplens = None
        self.epcount = 0
        self.tstart = time.time()

    def reset(self):
        obs = self.venv.reset()
        self.eprets = np.zeros(self.num_envs, 'f')
        self.eplens = np.zeros(self.num_envs, 'i')
        return obs

    def step_wait(self):
        obs, rews, dones, infos = self.venv.step_wait()
        self.eprets += rews
        self.eplens += 1

        newinfos = list(infos[:])
        for i in range(len(dones)):
            if dones[i]:
                info = infos[i].copy()
                ret = self.eprets[i]
                eplen = self.eplens[i]
                epinfo = {'r': ret, 'l': eplen, 't': round(time.time() - self.tstart, 6)}
                info['episode'] = epinfo
                self.epcount += 1
                self.eprets[i] = 0
                self.eplens[i] = 0
                newinfos[i] = info
        return obs, rews, dones, newinfos

class MicroRTSStatsRecorder(VecEnvWrapper):
    def __init__(self, env, gamma):
        super().__init__(env)
        self.gamma = gamma

    def reset(self):
        obs = self.venv.reset()
        self.raw_rewards = [[] for _ in range(self.num_envs)]
        return obs

    def step_wait(self):
        obs, rews, dones, infos = self.venv.step_wait()
        for i in range(len(dones)):
            self.raw_rewards[i] += [infos[i]["raw_rewards"]]
        newinfos = list(infos[:])
        for i in range(len(dones)):
            if dones[i]:
                info = infos[i].copy()
                raw_rewards = np.array(self.raw_rewards[i]).sum(0)
                raw_names = [str(rf) for rf in self.rfs]
                info['microrts_stats'] = dict(zip(raw_names, raw_rewards))
                self.raw_rewards[i] = []
                newinfos[i] = info
        return obs, rews, dones, newinfos

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SAC with 2 Q functions, Online updates')
    # Common arguments
    parser.add_argument('--exp-name', type=str, default=os.path.basename(__file__).rstrip(".py"),
                        help='the name of this experiment')
    parser.add_argument('--gym-id', type=str, default="MiniGrid-DoorKey-5x5-v0",
                        help='the id of the gym environment')
    parser.add_argument('--learning-rate', type=float, default=3e-4,
                        help='the learning rate of the optimizer')
    parser.add_argument('--seed', type=int, default=2,
                        help='seed of the experiment')
    parser.add_argument('--episode-length', type=int, default=0,
                        help='the maximum length of each episode')
    parser.add_argument('--total-timesteps', type=int, default=20000000,
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
    parser.add_argument('--learning-starts', type=int, default=0,
                        help="timestep to start learning")
    parser.add_argument('--n-step', type=int, default=6,
                        help="n step")
    parser.add_argument('--num-bot-envs', type=int, default=1,
                        help='the number of bot game environment; 16 bot envs measn 16 games')
    parser.add_argument('--num-selfplay-envs', type=int, default=0,
                        help='the number of self play envs; 16 self play envs means 8 games')


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
env = MicroRTSGridModeVecEnv(
    num_selfplay_envs=args.num_selfplay_envs,
    num_bot_envs=args.num_bot_envs,
    max_steps=1200,
    render_theme=2,
    ai2s=[microrts_ai.workerRushAI for _ in range(args.num_bot_envs)],
    map_path="maps/8x8/basesWorkers8x8.xml",
    reward_weight=np.array([10.0, 1.0, 1.0, 0.2, 1.0, 4.0])
)
mapsize=8*8
env = MicroRTSStatsRecorder(env, args.gamma)
env = VecMonitor(env)
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
class CategoricalMasked(Categorical):
    def __init__(self, probs=None, logits=None, validate_args=None, masks=[], sw=None):
        self.masks = masks
        if len(self.masks) == 0:
            super(CategoricalMasked, self).__init__(probs, logits, validate_args)
        else:
            self.masks = masks.bool()
            logits = torch.where(self.masks, logits, torch.tensor(-1e+8, device=device))
            super(CategoricalMasked, self).__init__(probs, logits, validate_args)

    def entropy(self):
        if len(self.masks) == 0:
            return super(CategoricalMasked, self).entropy()
        p_log_p = self.logits * self.probs
        p_log_p = torch.where(self.masks, p_log_p, torch.tensor(0.).to(device))
        return -p_log_p.sum(-1)

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Policy(nn.Module):
    def __init__(self, mapsize=8*8):
        super(Policy, self).__init__()
        self.mapsize = mapsize
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(27, 16, kernel_size=3, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(16, 32, kernel_size=2)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(128, 128)),
            nn.ReLU())

        self.logits = layer_init(nn.Linear(128, self.mapsize * env.action_space.nvec[1:].sum()), std=0.01)

    def forward(self, x):
        x = torch.Tensor(x).to(device)
        x = self.network(x.permute((0, 3, 1, 2)))
        logits = self.logits(x)
        probs = F.softmax(logits, -1)
        z = probs == 0.0
        z = z.float() * 1e-8
        log_probs = torch.log(probs + z)
        split_log_probs = log_probs.view(-1, env.action_space.nvec[1:].sum()).split(env.action_space.nvec[1:].tolist(), dim=1)
        split_probs = probs.view(-1, env.action_space.nvec[1:].sum()).split(env.action_space.nvec[1:].tolist(), dim=1)
        return split_probs, split_log_probs

    def get_action(self, x, envs=None):
        split_logits, _ = self.forward(x)

        invalid_action_masks = torch.tensor(np.array(envs.vec_client.getMasks(0))).to(device)
        invalid_action_masks = invalid_action_masks.view(-1, invalid_action_masks.shape[-1])
        split_invalid_action_masks = torch.split(invalid_action_masks[:, 1:], envs.action_space.nvec[1:].tolist(),
                                                 dim=1)
        multi_categoricals = [CategoricalMasked(logits=logits, masks=iam) for (logits, iam) in
                              zip(split_logits, split_invalid_action_masks)]
        action = torch.stack([categorical.sample() for categorical in multi_categoricals])
        num_predicted_parameters = len(envs.action_space.nvec) - 1
        action = action.T.view(-1, 64, num_predicted_parameters)
        invalid_action_masks = invalid_action_masks.view(-1, 64, envs.action_space.nvec[1:].sum() + 1)
        return action, invalid_action_masks


class SoftQNetwork(nn.Module):
    def __init__(self, mapsize=8 * 8):
        super(SoftQNetwork, self).__init__()

        self.network = nn.Sequential(
            layer_init(nn.Conv2d(27, 16, kernel_size=3, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(16, 32, kernel_size=2)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(128, 128)),
            nn.ReLU())

        self.q_value = layer_init(nn.Linear(128, mapsize * env.action_space.nvec[1:].sum()), std=1)

    def forward(self, x):
        x = torch.Tensor(x).to(device)
        x = self.network(x.permute((0, 3, 1, 2)))
        x = self.q_value(x)
        x = torch.split(x.view(-1, env.action_space.nvec[1:].sum()), env.action_space.nvec[1:].tolist(), dim=1)
        return x

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
        _, _, reward, next_observation, done, invalid_action_mask = self.n_step_buffer[-1]
        for _, _, r, next_obs, do, invalid_action_mask in reversed(list(self.n_step_buffer)[:-1]):
            reward = self.gamma * reward * (1 - do) + r
            mext_observation, done = (next_obs, do) if do else (next_observation, done)
        return reward, next_observation, done

    def append(self, obs, action, reward, next_obs, done, invalid_action_mask):
        self.n_step_buffer.append((obs, action, reward, next_obs, done, invalid_action_mask))
        if len(self.n_step_buffer) < self.n_step:
            return
        reward, next_obs, done = self.get_n_step()
        obs, action, _, _, _, invalid_action_mask = self.n_step_buffer[0]
        self._storage.append([obs, action, reward, next_obs, done, invalid_action_mask])

    def sample(self, batch_size):
        idxes = np.random.choice(len(self._storage), batch_size, replace=True)
        obses_t, actions, rewards, obses_tp1, dones, invalid_action_masks = [], [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, action, reward, obs_tp1, done, invalid_action_mask = data
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            dones.append(done)
            invalid_action_masks.append(invalid_action_mask)
        return np.array(obses_t), np.array(actions), np.array(rewards), np.array(obses_tp1), np.array(dones), np.array(invalid_action_masks)

rb=ReplayBufferNStep(args.buffer_size, args.n_step, args.gamma)

pg = Policy().to(device)
qf1 = SoftQNetwork().to(device)
qf2 = SoftQNetwork().to(device)
qf1_target = SoftQNetwork().to(device)
qf2_target = SoftQNetwork().to(device)
qf1_target.eval()
qf2_target.eval()
qf1_target.load_state_dict(qf1.state_dict())
qf2_target.load_state_dict(qf2.state_dict())
values_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
policy_optimizer = optim.Adam(list(pg.parameters()), lr=args.policy_lr)
loss_fn = nn.MSELoss()

# Automatic entropy tuning
if args.autotune:
    target_entropy = 0.98 * (-np.log(1 / 8*8 * env.action_space.nvec[1:].sum()))
    log_alpha = torch.zeros(1, requires_grad=True, device=device)
    alpha = log_alpha.exp().item()
    a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
else:
    alpha = args.alpha

# TRY NOT TO MODIFY: start the game
start_time = time.time()
global_episode = 0
obs, done = env.reset(), False
for global_step in range(1, args.total_timesteps+1):
    # ALGO LOGIC: put action logic here
    env.render()
    action, invalid_action_mask = pg.get_action(obs, envs=env)
    # TRY NOT TO MODIFY: execute the game and log data.
    real_action = torch.cat([
        torch.stack(
            [torch.arange(0, mapsize, device=device) for i in range(env.num_envs)
             ]).unsqueeze(2), action], 2)

    # at this point, the `real_action` has shape (num_envs, map_height*map_width, 8)
    # so as to predict an action for each cell in the map; this obviously include a
    # lot of invalid actions at cells for which no source units exist, so the rest of
    # the code removes these invalid actions to speed things up
    real_action = real_action.cpu().numpy()
    valid_actions = real_action[invalid_action_mask[:, :, 0].bool().cpu().numpy()]
    valid_actions_counts = invalid_action_mask[:, :, 0].sum(1).long().cpu().numpy()
    java_valid_actions = []
    valid_action_idx = 0
    for env_idx, valid_action_count in enumerate(valid_actions_counts):
        java_valid_action = []
        for c in range(valid_action_count):
            java_valid_action += [JArray(JInt)(valid_actions[valid_action_idx])]
            valid_action_idx += 1
        java_valid_actions += [JArray(JArray(JInt))(java_valid_action)]
    java_valid_actions = JArray(JArray(JArray(JInt)))(java_valid_actions)

    try:
        next_obs, reward, done, infos = env.step(java_valid_actions)
    except Exception as e:
        e.printStackTrace()
        raise

    rb.append(obs.squeeze(), action.cpu().numpy().squeeze(), reward, next_obs.squeeze(), done, invalid_action_mask.cpu().numpy())
    obs = np.array(next_obs)

    for info in infos:
        if 'episode' in info.keys():
            print(f"global_step={global_step}, episode_reward={info['episode']['r']}")
            writer.add_scalar("charts/episode_reward", info['episode']['r'], global_step)
            for key in info['microrts_stats']:
                writer.add_scalar(f"charts/episode_reward/{key}", info['microrts_stats'][key], global_step)
            break

    # ALGO LOGIC: training.
    if len(rb) > args.batch_size and global_step % 4 == 0: # starts update as soon as there is enough data.
        s_obs, s_actions, s_rewards, s_next_obses, s_dones, invalid_action_mask = rb.sample(args.batch_size)
        with torch.no_grad():
            probs_split, next_state_log_probs_split = pg.forward(s_next_obses)

            qf1_next_target_split = qf1_target.forward(s_next_obses)
            qf2_next_target_split = qf2_target.forward(s_next_obses)
            min_target = tuple([torch.min(qf1, qf2) for qf1, qf2 in zip(qf1_next_target_split, qf2_next_target_split)])
            #min_qf_next_target = (probs * (torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_probs)).sum(-1)
            min_next_target = tuple([(p * (m - alpha*n)).sum(-1) for p, m, n in zip(probs_split, min_target, next_state_log_probs_split)])

            next_q_value = torch.stack([torch.Tensor(s_rewards).to(device) + (1 - torch.Tensor(s_dones).to(device)) * (args.gamma ** args.n_step) * m.view(args.batch_size, -1) for m in min_next_target])
            #next_q_value = torch.Tensor(s_rewards).to(device) + (1 - torch.Tensor(s_dones).to(device)) * (args.gamma ** args.n_step) * min_next_target

        s_actions = torch.LongTensor(s_actions).to(device).view(-1, s_actions.shape[-1]).T
        qf1_a_values = qf1.forward(s_obs)
        qf2_a_values = qf2.forward(s_obs)


        qf1_a_values = torch.stack([values.gather(1, a.view(-1, 1)).view(-1) for a, values in zip(s_actions, qf1_a_values)])
        qf2_a_values = torch.stack([values.gather(1, a.view(-1, 1)).view(-1) for a, values in zip(s_actions, qf2_a_values)])

        #qf1_a_values = qf1.forward(s_obs)[np.arange(args.batch_size), np.array(s_actions)]
        #qf2_a_values = qf2.forward(s_obs)[np.arange(args.batch_size), np.array(s_actions)]
        qf1_loss = loss_fn(qf1_a_values, next_q_value.view(7, -1))
        qf2_loss = loss_fn(qf2_a_values, next_q_value.view(7, -1))
        qf_loss = (qf1_loss + qf2_loss) / 2

        values_optimizer.zero_grad()
        qf_loss.backward()
        values_optimizer.step()

        if global_step % args.policy_frequency == 0: # TD 3 Delayed update support
            for _ in range(args.policy_frequency): # compensate for the delay by doing 'actor_update_interval' instead of 1
                probs_split, log_probs_split = pg.forward(s_obs)

                qf1_pi_split = qf1.forward(s_obs)
                qf2_pi_split = qf2.forward(s_obs)

                min_qf_pi = tuple([torch.min(qf1, qf2) for qf1, qf2 in zip(qf1_pi_split, qf2_pi_split)])
                #min_qf_pi = torch.min(qf1_pi, qf2_pi)
                #policy_loss = (probs * (alpha * log_probs - min_qf_pi)).sum(-1).mean()
                policy_loss = torch.stack([(p * (alpha * l_p - min)).sum(-1).mean() for p, l_p, min in zip(probs_split, log_probs_split, min_qf_pi)]).sum()

                policy_optimizer.zero_grad()
                policy_loss.backward()
                policy_optimizer.step()

                if args.autotune:
                    with torch.no_grad():
                        probs, log_probs = pg.forward(s_obs)
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
    writer.add_scalar("charts/sps", int(global_step / (time.time() - start_time)), global_step)
    if done:
        # Reseting what need to be
        obs, done = env.reset(), False


writer.close()
env.close()
