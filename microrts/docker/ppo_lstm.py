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
import gym_microrts
from gym.wrappers import TimeLimit, Monitor
from gym_microrts.envs.vec_env import MicroRTSVecEnv
from gym_microrts import microrts_ai
from gym.spaces import Discrete, Box, MultiBinary, MultiDiscrete, Space
import time
import random
import os
from stable_baselines3.common.vec_env import VecEnvWrapper, VecVideoRecorder

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PPO agent')
    # Common arguments
    parser.add_argument('--exp-name', type=str, default=os.path.basename(__file__).rstrip(".py"),
                        help='the name of this experiment')
    parser.add_argument('--gym-id', type=str, default="Microrts10-workerRushAI-lstm",
                        help='the id of the gym environment')
    parser.add_argument('--learning-rate', type=float, default=2.5e-4,
                        help='the learning rate of the optimizer')
    parser.add_argument('--seed', type=int, default=1,
                        help='seed of the experiment')
    parser.add_argument('--total-timesteps', type=int, default=50000000,
                        help='total timesteps of the experiments')
    parser.add_argument('--torch-deterministic', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True,
                        help='if toggled, `torch.backends.cudnn.deterministic=False`')
    parser.add_argument('--cuda', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True,
                        help='if toggled, cuda will not be enabled by default')
    parser.add_argument('--prod-mode', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True,
                        help='run the script in production mode and use wandb to log outputs')
    parser.add_argument('--capture-video', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True,
                        help='weather to capture videos of the agent performances (check out `videos` folder)')
    parser.add_argument('--wandb-project-name', type=str, default="microrts",
                        help="the wandb's project name")
    parser.add_argument('--wandb-entity', type=str, default=None,
                        help="the entity (team) of wandb's project")

    # Algorithm specific arguments
    parser.add_argument('--n-minibatch', type=int, default=4,
                        help='the number of mini batch')
    parser.add_argument('--num-envs', type=int, default=16,
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
    parser.add_argument('--kle-stop', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True,
                        help='If toggled, the policy updates will be early stopped w.r.t target-kl')
    parser.add_argument('--kle-rollback', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True,
                        help='If toggled, the policy updates will roll back to previous policy if KL exceeds target-kl')
    parser.add_argument('--target-kl', type=float, default=0.03,
                        help='the target-kl variable that is referred by --kl')
    parser.add_argument('--gae', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True,
                        help='Use GAE for advantage computation')
    parser.add_argument('--norm-adv', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True,
                        help="Toggles advantages normalization")
    parser.add_argument('--anneal-lr', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True,
                        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument('--clip-vloss', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True,
                        help='Toggles wheter or not to use a clipped loss for the value function, as per the paper.')
    parser.add_argument('--rnn-hidden-size', type=int, default=256,
                        help='rnn hidden size')

    args = parser.parse_args()
    if not args.seed:
        args.seed = int(time.time())

args.batch_size = int(args.num_envs * args.num_steps)
args.minibatch_size = int(args.batch_size // args.n_minibatch)

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

# TRY NOT TO MODIFY: setup the environment
experiment_name = f"{args.gym_id}__{args.exp_name}__{args.seed}__" + time.strftime("%d-%m-%Y_%H-%M-%S")
writer = SummaryWriter(f"runs/{experiment_name}")
writer.add_text('hyperparameters', "|param|value|\n|-|-|\n%s" % (
    '\n'.join([f"|{key}|{value}|" for key, value in vars(args).items()])))
if args.prod_mode:
    import wandb

    run = wandb.init(project=args.wandb_project_name, entity=args.wandb_entity, sync_tensorboard=True,
                     config=vars(args), name=experiment_name, monitor_gym=True, save_code=True)
    writer = SummaryWriter(f"/tmp/{experiment_name}")

# TRY NOT TO MODIFY: seeding
device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = args.torch_deterministic

envs = MicroRTSVecEnv(
    num_envs=args.num_envs,
    max_steps=20000,
    render_theme=2,
    ai2s=[microrts_ai.workerRushAI for _ in range(args.num_envs)],
    map_path="maps/10x10/basesWorkers10x10.xml",
    reward_weight=np.array([10.0, 1.0, 1.0, 0.2, 1.0, 4.0])
)
envs = MicroRTSStatsRecorder(envs, args.gamma)
envs = VecMonitor(envs)
envs = VecPyTorch(envs, device)
if args.capture_video:
    envs = VecVideoRecorder(envs, f'videos/{experiment_name}',
                            record_video_trigger=lambda x: x % 1000000 == 0, video_length=2000)
assert isinstance(envs.action_space, MultiDiscrete), "only MultiDiscrete action space is supported"

# ALGO LOGIC: initialize agent here:
class CategoricalMasked(Categorical):
    def __init__(self, probs=None, logits=None, validate_args=None, masks=[]):
        self.masks = masks
        if len(self.masks) == 0:
            super(CategoricalMasked, self).__init__(probs, logits, validate_args)
        else:
            self.masks = masks.type(torch.BoolTensor).to(device)
            logits = torch.where(self.masks, logits, torch.tensor(-1e+8).to(device))
            super(CategoricalMasked, self).__init__(probs, logits, validate_args)

    def entropy(self):
        if len(self.masks) == 0:
            return super(CategoricalMasked, self).entropy()
        p_log_p = self.logits * self.probs
        p_log_p = torch.where(self.masks, p_log_p, torch.tensor(0.).to(device))
        return -p_log_p.sum(-1)


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

def recurrent_generator(obs, logprobs, actions, advantages, returns, values, rnn_hidden_states, masks, invalid_action_masks):
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
        a_invalid_action_masks = []

        for offset in range(num_envs_per_batch):
            ind = perm[start_ind + offset]
            a_rnn_hidden_states.append(torch.stack((rnn_hidden_states[0:1, 0, ind], rnn_hidden_states[0:1, 1, ind])))
            a_obs.append(obs[:, ind])
            a_actions.append(actions[:, ind])
            a_values.append(values[:, ind])
            a_returns.append(returns[:, ind])
            a_masks.append(masks[:, ind])
            a_logprobs.append(logprobs[:, ind])
            a_advantages.append(advantages[:, ind])
            a_invalid_action_masks.append(invalid_action_masks[:, ind])

        T, N = args.num_steps, num_envs_per_batch

        b_rnn_hidden_states = torch.stack(a_rnn_hidden_states, 1).view(N, 2, -1)
        b_obs = _flatten_helper(T, N, torch.stack(a_obs, 1))
        b_actions = _flatten_helper(T, N, torch.stack(a_actions, 1))
        b_values = _flatten_helper(T, N, torch.stack(a_values, 1))
        b_return = _flatten_helper(T, N, torch.stack(a_returns, 1))
        b_masks = _flatten_helper(T, N, torch.stack(a_masks, 1))
        b_logprobs = _flatten_helper(T, N, torch.stack(a_logprobs, 1))
        b_advantages = _flatten_helper(T, N, torch.stack(a_advantages, 1))
        b_invalid_action_masks = _flatten_helper(T, N, torch.stack(a_invalid_action_masks, 1))

        yield b_obs, b_rnn_hidden_states, b_actions, \
              b_values, b_return, b_masks, b_logprobs, b_advantages, b_invalid_action_masks


class Agent(nn.Module):
    def __init__(self, frames=4):
        super(Agent, self).__init__()
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(27, 10, kernel_size=3, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(10, 20, kernel_size=2)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(180, 180)),
            nn.ReLU())

        self.lstm = nn.LSTM(180, args.rnn_hidden_size)
        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                nn.init.orthogonal_(param, np.sqrt(2))
        self.actor = layer_init(nn.Linear(args.rnn_hidden_size, envs.action_space.nvec.sum()), std=0.01)
        self.critic = layer_init(nn.Linear(args.rnn_hidden_size, 1), std=1)

    def forward(self, x, hxs, mask):
        x = self.network(x.permute((0, 3, 1, 2)))
        hs = hxs[0]
        cs = hxs[1]
        if x.size(0) == hs.size(0):
            x, (hs, cs) = self.lstm(x.unsqueeze(0), ((hs * mask).unsqueeze(0), (cs * mask).unsqueeze(0)))
            x = x.squeeze()
        else:
            N = hs.size(0)
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
            hs = hs.unsqueeze(0)
            cs = cs.unsqueeze(0)
            outputs = []
            for i in range(len(has_zeros) - 1):
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]

                rnn_scores, (hs, cs) = self.lstm(
                    x[start_idx:end_idx],
                    (hs * masks[start_idx].view(1, -1, 1), cs * masks[start_idx].view(1, -1, 1)))

                outputs.append(rnn_scores)

            x = torch.cat(outputs, dim=0)
            x = x.view(T * N, -1)
        hs = hs.squeeze(0)
        cs = cs.squeeze(0)
        hxs = torch.stack([hs, cs])
        return x, hxs

    def get_action(self, x, rnn_hidden_state, mask, action=None, invalid_action_masks=None, envs=None):
        x, rnn_hidden_state = self.forward(x, rnn_hidden_state, mask)
        logits = self.actor(x)
        split_logits = torch.split(logits, envs.action_space.nvec.tolist(), dim=1)

        if action is None:
            # 1. select source unit based on source unit mask
            source_unit_mask = torch.Tensor(np.array(envs.vec_client.getUnitLocationMasks()).reshape(args.num_envs, -1))
            multi_categoricals = [CategoricalMasked(logits=split_logits[0], masks=source_unit_mask)]
            action_components = [multi_categoricals[0].sample()]
            # 2. select action type and parameter section based on the
            #    source-unit mask of action type and parameters
            # print(np.array(envs.vec_client.getUnitActionMasks(action_components[0].cpu().numpy())).reshape(args.num_envs, -1))
            source_unit_action_mask = torch.Tensor(
                np.array(envs.vec_client.getUnitActionMasks(action_components[0].cpu().numpy())).reshape(args.num_envs,
                                                                                                         -1))
            split_suam = torch.split(source_unit_action_mask, envs.action_space.nvec.tolist()[1:], dim=1)
            multi_categoricals = multi_categoricals + [CategoricalMasked(logits=logits, masks=iam) for (logits, iam) in
                                                       zip(split_logits[1:], split_suam)]
            invalid_action_masks = torch.cat((source_unit_mask, source_unit_action_mask), 1)
            action_components += [categorical.sample() for categorical in multi_categoricals[1:]]
            action = torch.stack(action_components)
        else:
            split_invalid_action_masks = torch.split(invalid_action_masks, envs.action_space.nvec.tolist(), dim=1)
            multi_categoricals = [CategoricalMasked(logits=logits, masks=iam) for (logits, iam) in
                                  zip(split_logits, split_invalid_action_masks)]
        logprob = torch.stack([categorical.log_prob(a) for a, categorical in zip(action, multi_categoricals)])
        entropy = torch.stack([categorical.entropy() for categorical in multi_categoricals])
        return rnn_hidden_state, action, logprob.sum(0), entropy.sum(0), invalid_action_masks

    def get_value(self, x, rnn_hidden_state, mask):
        x, rnn_hidden_state = self.forward(x, rnn_hidden_state, mask)
        return self.critic(x)


agent = Agent().to(device)
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
values = torch.zeros((args.num_steps, args.num_envs)).to(device)
invalid_action_masks = torch.zeros((args.num_steps, args.num_envs) + (envs.action_space.nvec.sum(),)).to(device)
rnn_hidden_states = torch.zeros((args.num_steps, 2, args.num_envs, args.rnn_hidden_size)).to(device)
masks = torch.ones((args.num_steps, args.num_envs, 1)).to(device)

# TRY NOT TO MODIFY: start the game
global_step = 0
start_time = time.time()
# Note how `next_obs` and `next_done` are used; their usage is equivalent to
# https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/84a7582477fb0d5c82ad6d850fe476829dddd2e1/a2c_ppo_acktr/storage.py#L60
next_obs = envs.reset()
next_done = torch.zeros(args.num_envs).to(device)
num_updates = args.total_timesteps // args.batch_size
mask = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in next_done]).to(device)
rnn_hidden_state = torch.zeros((2, args.num_envs, args.rnn_hidden_size)).to(device)
## CRASH AND RESUME LOGIC:
starting_update = 1
if args.prod_mode and wandb.run.resumed:
    print("previous run.summary", run.summary)
    starting_update = run.summary['charts/update'] + 1
    global_step = starting_update * args.batch_size
    api = wandb.Api()
    run = api.run(run.get_url()[len("https://app.wandb.ai/"):])
    model = run.file('agent.pt')
    model.download(f"models/{experiment_name}/")
    agent.load_state_dict(torch.load(f"models/{experiment_name}/agent.pt"))
    agent.eval()
    print(f"resumed at update {starting_update}")
for update in range(starting_update, num_updates + 1):
    # Annealing the rate if instructed to do so.
    if args.anneal_lr:
        frac = 1.0 - (update - 1.0) / num_updates
        lrnow = lr(frac)
        optimizer.param_groups[0]['lr'] = lrnow

    # TRY NOT TO MODIFY: prepare the execution of the game.
    for step in range(0, args.num_steps):
        envs.render()
        global_step += 1 * args.num_envs
        obs[step] = next_obs
        dones[step] = next_done
        rnn_hidden_states[step] = rnn_hidden_state
        masks[step] = mask

        # ALGO LOGIC: put action logic here
        with torch.no_grad():
            values[step] = agent.get_value(obs[step], rnn_hidden_state, mask).flatten()
            rnn_hidden_state, action, logproba, _, invalid_action_masks[step] = agent.get_action(obs[step], rnn_hidden_state, mask, envs=envs)

        actions[step] = action.T
        logprobs[step] = logproba

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rs, ds, infos = envs.step(action.T)
        rewards[step], next_done = rs.view(-1), torch.Tensor(ds).to(device)
        mask = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in next_done]).to(device)

        for info in infos:
            if 'episode' in info.keys():
                print(f"global_step={global_step}, episode_reward={info['episode']['r']}")
                writer.add_scalar("charts/episode_reward", info['episode']['r'], global_step)
                for key in info['microrts_stats']:
                    writer.add_scalar(f"charts/episode_reward/{key}", info['microrts_stats'][key], global_step)
                break

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
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
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
                    nextnonterminal = 1.0 - dones[t + 1]
                    next_return = returns[t + 1]
                returns[t] = rewards[t] + args.gamma * nextnonterminal * next_return
            advantages = returns - values

    # Optimizaing the policy and value network
    target_agent = Agent().to(device)
    inds = np.arange(args.batch_size, )
    for i_epoch_pi in range(args.update_epochs):
        target_agent.load_state_dict(agent.state_dict())
        data_generator = recurrent_generator(obs, logprobs, actions, advantages, returns, values,
                                             rnn_hidden_states, masks, invalid_action_masks)
        for batch in data_generator:
            b_obs, b_rnn_hidden_states, b_actions, b_values, b_returns, b_masks, b_logprobs, b_advantages, b_invalid_action_masks = batch

            if args.norm_adv:
                b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

            _, _, newlogproba, entropy, _ = agent.get_action(
                b_obs,
                b_rnn_hidden_states, b_masks,
                b_actions.long().T,
                b_invalid_action_masks,
                envs)
            ratio = (newlogproba - b_logprobs).exp()

            # Stats
            approx_kl = (b_logprobs - newlogproba).mean()

            # Policy loss
            pg_loss1 = -b_advantages * ratio
            pg_loss2 = -b_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()
            entropy_loss = entropy.mean()

            # Value loss
            new_values = agent.get_value(b_obs, b_rnn_hidden_states, b_masks).view(-1)
            if args.clip_vloss:
                v_loss_unclipped = ((new_values - b_returns) ** 2)
                v_clipped = b_values + torch.clamp(new_values - b_values, -args.clip_coef,
                                                                  args.clip_coef)
                v_loss_clipped = (v_clipped - b_returns) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()
            else:
                v_loss = 0.5 * ((new_values - b_returns) ** 2)

            loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
            optimizer.step()

        if args.kle_stop:
            if approx_kl > args.target_kl:
                break
        if args.kle_rollback:
            if (b_logprobs - agent.get_action(
                    b_obs,
                    b_actions.long().T,
                    b_invalid_action_masks,
                    envs)[1]).mean() > args.target_kl:
                agent.load_state_dict(target_agent.state_dict())
                break

    ## CRASH AND RESUME LOGIC:
    if args.prod_mode:
        if not os.path.exists(f"models/{experiment_name}"):
            os.makedirs(f"models/{experiment_name}")
        torch.save(agent.state_dict(), f"{wandb.run.dir}/agent.pt")
        wandb.save(f"agent.pt")

    # TRY NOT TO MODIFY: record rewards for plotting purposes
    writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]['lr'], global_step)
    writer.add_scalar("charts/update", update, global_step)
    writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
    writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
    writer.add_scalar("losses/entropy", entropy.mean().item(), global_step)
    writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
    if args.kle_stop or args.kle_rollback:
        writer.add_scalar("debug/pg_stop_iter", i_epoch_pi, global_step)
    print("SPS:", int(global_step / (time.time() - start_time)))

envs.close()
writer.close()
