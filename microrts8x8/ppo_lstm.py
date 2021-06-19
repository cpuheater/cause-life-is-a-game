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
    parser.add_argument('--learning-rate', type=float, default=3e-4,
                        help='the learning rate of the optimizer')
    parser.add_argument('--seed', type=int, default=1,
                        help='seed of the experiment')
    parser.add_argument('--total-timesteps', type=int, default=17000000,
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
    parser.add_argument('--num-envs', type=int, default=24,
                        help='the number of parallel game environment')
    parser.add_argument('--num-steps', type=int, default=256,
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
    parser.add_argument('--rnn-hidden-size', type=int, default=512,
                        help='rnn hidden size')
    parser.add_argument('--seq-length', type=int, default=16,
                        help='seq length')

    args = parser.parse_args()
    #if not args.seed:
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
experiment_name = f"{args.gym_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
writer = SummaryWriter(f"runs/{experiment_name}")
writer.add_text('hyperparameters', "|param|value|\n|-|-|\n%s" % (
    '\n'.join([f"|{key}|{value}|" for key, value in vars(args).items()])))
if args.prod_mode:
    import wandb
    run = wandb.init(
        project=args.wandb_project_name, entity=args.wandb_entity,
        # sync_tensorboard=True,
        config=vars(args), name=experiment_name, monitor_gym=True, save_code=True)
    wandb.tensorboard.patch(save=False)
    writer = SummaryWriter(f"/tmp/{experiment_name}")
    CHECKPOINT_FREQUENCY = 50

# TRY NOT TO MODIFY: seeding
device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = args.torch_deterministic
envs = MicroRTSVecEnv(
    num_envs=args.num_envs,
    max_steps=2000,
    render_theme=2,
    ai2s=[microrts_ai.workerRushAI for _ in range(args.num_envs)],
    map_path="maps/8x8/basesWorkers8x8.xml",
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

def masked_mean(tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (tensor.T * mask).sum() / torch.clamp((torch.ones_like(tensor.T) * mask).float().sum(), min=1.0)

def recurrent_generator(episode_done_indices, obs, actions, logprobs, values, advantages, returns, rnn_hidden_states, cell_hidden_states, invalid_action_masks):

    # Supply training samples
    samples = {
        'vis_obs': obs.permute(1, 0, 2, 3, 4).cpu().numpy(),
        'actions': actions.permute(1, 0, 2).cpu().numpy(),
        'values': values.permute(1, 0).cpu().numpy(),
        'log_probs': logprobs.permute(1, 0).cpu().numpy(),
        'advantages': advantages.permute(1, 0).cpu().numpy(),
        'returns': returns.permute(1, 0).cpu().numpy(),
        'loss_mask': np.ones((args.num_envs, args.num_steps), dtype=np.float32),
        "hxs": rnn_hidden_states.permute(1, 0, 2).cpu().numpy(),
        "cxs": cell_hidden_states.permute(1, 0, 2).cpu().numpy(),
        "invalid_action_masks": invalid_action_masks.permute(1, 0, 2).cpu().numpy()
    }

    max_sequence_length = 1
    for w in range(args.num_envs):
        if len(episode_done_indices[w]) == 0 or episode_done_indices[w][-1] != args.num_steps - 1:
            episode_done_indices[w].append(args.num_steps - 1)

    for key, value in samples.items():
        sequences = []
        for w in range(args.num_envs):
            start_index = 0
            for done_index in episode_done_indices[w]:
                episode = value[w, start_index:done_index + 1]
                start_index = done_index + 1
                if args.seq_length > 0:
                    for start in range(0, len(episode), args.seq_length): # step min seq_length
                        end = start + args.seq_length
                        sequences.append(episode[start:end])
                    max_sequence_length = args.seq_length
                else:
                    sequences.append(episode)
                    max_sequence_length = len(episode) if len(
                        episode) > max_sequence_length else max_sequence_length

        for i, sequence in enumerate(sequences):
            sequences[i] =  pad_sequence(sequence, max_sequence_length)

        samples[key] = np.stack(sequences, axis=0)
        if (key == "hxs" or key == "cxs"):
            samples[key] = samples[key][:, 0]

    num_sequences = len(samples["values"])
    actual_sequence_length = max_sequence_length

    samples_flat = {}
    for key, value in samples.items():
        if (not key == "hxs" and not key == "cxs"):
            value = value.reshape(value.shape[0] * value.shape[1], *value.shape[2:])
        samples_flat[key] = torch.tensor(value, dtype=torch.float32, device=device)


    num_sequences_per_batch = num_sequences // args.n_minibatch
    num_sequences_per_batch = [
                                  num_sequences_per_batch] * args.n_minibatch
    remainder = num_sequences % args.n_minibatch
    for i in range(remainder):
        num_sequences_per_batch[i] += 1
    indices = np.arange(0, num_sequences * actual_sequence_length).reshape(num_sequences, actual_sequence_length)
    sequence_indices = torch.randperm(num_sequences)

    start = 0
    for num_sequences in num_sequences_per_batch:
        end = start + num_sequences
        mini_batch_indices = indices[sequence_indices[start:end]].reshape(-1)
        mini_batch = {}
        for key, value in samples_flat.items():
            if key != "hxs" and key != "cxs":
                mini_batch[key] = value[mini_batch_indices].to(device)
            else:
                mini_batch[key] = value[sequence_indices[start:end]].to(device)
        start = end
        yield mini_batch

def pad_sequence(sequence, target_length):
    if isinstance(sequence, torch.Tensor):
        sequence = sequence.numpy()
    delta_length = target_length - len(sequence)
    if delta_length <= 0:
        return sequence
    if len(sequence.shape) > 1:
        padding = np.zeros(((delta_length,) + sequence.shape[1:]), dtype=sequence.dtype)
    else:
        padding = np.zeros(delta_length, dtype=sequence.dtype)
    return np.concatenate((sequence, padding), axis=0)

class Agent(nn.Module):
    def __init__(self, frames=4):
        super(Agent, self).__init__()
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(27, 16, kernel_size=3, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(16, 32, kernel_size=2)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(128, args.rnn_hidden_size)),
            nn.ReLU())
        self.rnn = nn.LSTM(args.rnn_hidden_size, args.rnn_hidden_size, batch_first=True)
        for name, param in self.rnn.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                nn.init.orthogonal_(param, np.sqrt(2))
        self.lin_hidden = layer_init(nn.Linear(args.rnn_hidden_size, args.rnn_hidden_size))
        self.lin_value = layer_init(nn.Linear(args.rnn_hidden_size, args.rnn_hidden_size))
        self.lin_policy = layer_init(nn.Linear(args.rnn_hidden_size, args.rnn_hidden_size))
        self.actor = layer_init(nn.Linear(args.rnn_hidden_size, envs.action_space.nvec.sum()), std=0.01)
        self.critic = layer_init(nn.Linear(args.rnn_hidden_size, 1), std=1)
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, x, rnn_state, seq_length=1):
        x = self.network(x.permute((0, 3, 1, 2)))
        if seq_length == 1:
            x, rnn_state = self.rnn(x.unsqueeze(1), rnn_state)
            x = x.squeeze(1)
        else:
            x_shape = tuple(x.size())
            x = x.reshape((x_shape[0] // seq_length), seq_length, x_shape[1])
            x, rnn_state = self.rnn(x, rnn_state)
            x_shape = tuple(x.size())
            x = x.reshape(x_shape[0] * x_shape[1], x_shape[2])
        return x, rnn_state

    def get_action(self, x, rnn_state, seq_length=1, action=None, invalid_action_masks=None, envs=None):
        x, rnn_state = self.forward(x, rnn_state, seq_length)

        x = self.leaky_relu(self.lin_hidden(x))
        value = self.leaky_relu(self.lin_value(x))
        policy = self.leaky_relu(self.lin_policy(x))
        logits = self.actor(policy)

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
                np.array(envs.vec_client.getUnitActionMasks(action_components[0].cpu().numpy())).reshape(args.num_envs, -1))
            split_suam = torch.split(source_unit_action_mask, envs.action_space.nvec.tolist()[1:], dim=1)
            multi_categoricals = multi_categoricals + [CategoricalMasked(logits=logits, masks=iam) for (logits, iam) in zip(split_logits[1:], split_suam)]
            invalid_action_masks = torch.cat((source_unit_mask, source_unit_action_mask), 1)
            action_components += [categorical.sample() for categorical in multi_categoricals[1:]]
            action = torch.stack(action_components)
        else:
            split_invalid_action_masks = torch.split(invalid_action_masks, envs.action_space.nvec.tolist(), dim=1)
            multi_categoricals = [CategoricalMasked(logits=logits, masks=iam) for (logits, iam) in zip(split_logits, split_invalid_action_masks)]
        for a, categorical in zip(action, multi_categoricals):
            categorical.log_prob(a)
        logprob = torch.stack([categorical.log_prob(a) for a, categorical in zip(action, multi_categoricals)])
        entropy = torch.stack([categorical.entropy() for categorical in multi_categoricals])
        return self.critic(value), rnn_state, action, logprob.sum(0), entropy.sum(0), invalid_action_masks

    def get_value(self, x, rnn_state):
        x, rnn_state = self.forward(x, rnn_state)
        x = self.leaky_relu(self.lin_hidden(x))
        value = self.leaky_relu(self.lin_value(x))
        return self.critic(value)

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
rnn_hidden_states = torch.zeros((args.num_steps, args.num_envs, args.rnn_hidden_size)).to(device)
rnn_cell_states = torch.zeros((args.num_steps, args.num_envs, args.rnn_hidden_size)).to(device)
# TRY NOT TO MODIFY: start the game
global_step = 0
start_time = time.time()
# Note how `next_obs` and `next_done` are used; their usage is equivalent to
# https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/84a7582477fb0d5c82ad6d850fe476829dddd2e1/a2c_ppo_acktr/storage.py#L60
next_obs = envs.reset()
next_done = torch.zeros(args.num_envs).to(device)
rnn_hidden_state = torch.zeros((1, args.num_envs, args.rnn_hidden_size)).to(device)
rnn_cell_state = torch.zeros((1, args.num_envs, args.rnn_hidden_size)).to(device)
num_updates = args.total_timesteps // args.batch_size
## CRASH AND RESUME LOGIC:
starting_update = 1
if args.prod_mode and wandb.run.resumed:
    starting_update = run.summary.get('charts/update') + 1
    global_step = starting_update * args.batch_size
    api = wandb.Api()
    run = api.run(f"{run.entity}/{run.project}/{run.id}")
    model = run.file('agent.pt')
    model.download(f"models/{experiment_name}/")
    agent.load_state_dict(torch.load(f"models/{experiment_name}/agent.pt", map_location=device))
    agent.eval()
    print(f"resumed at update {starting_update}")
for update in range(starting_update, num_updates+1):
    # Annealing the rate if instructed to do so.
    if args.anneal_lr:
        frac = 1.0 - (update - 1.0) / num_updates
        lrnow = lr(frac)
        optimizer.param_groups[0]['lr'] = lrnow
    episode_done_indices = [[] for w in range(args.num_envs)]
    # TRY NOT TO MODIFY: prepare the execution of the game.
    for step in range(0, args.num_steps):
        envs.render()
        global_step += 1 * args.num_envs
        obs[step] = next_obs
        dones[step] = next_done
        rnn_hidden_states[step] = rnn_hidden_state
        rnn_cell_states[step] = rnn_cell_state
        # ALGO LOGIC: put action logic here
        with torch.no_grad():
            value, (rnn_hidden_state, rnn_cell_state), action, logproba, _, invalid_action_masks[step] = agent.get_action(obs[step], (rnn_hidden_state, rnn_cell_state), seq_length=1, envs=envs)

        actions[step] = action.T
        logprobs[step] = logproba
        values[step] = value.flatten()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rs, ds, infos = envs.step(action.T)
        rewards[step], next_done = rs.view(-1), torch.Tensor(ds).to(device)
        mask = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in next_done]).to(device)
        rnn_hidden_state = rnn_hidden_state * mask
        rnn_cell_state = rnn_cell_state * mask
        [episode_done_indices[index].append(step) for index in torch.nonzero(next_done).flatten().tolist()]
        for info in infos:
            if 'episode' in info.keys():
                print(f"global_step={global_step}, episode_reward={info['episode']['r']}")
                writer.add_scalar("charts/episode_reward", info['episode']['r'], global_step)
                for key in info['microrts_stats']:
                    writer.add_scalar(f"charts/episode_reward/{key}", info['microrts_stats'][key], global_step)
                break

    # bootstrap reward if not done. reached the batch limit
    with torch.no_grad():
        last_value = agent.get_value(next_obs.to(device), (rnn_hidden_state, rnn_cell_state)).reshape(1, -1)
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
    for i_epoch_pi in range(args.update_epochs):
        data_generator = recurrent_generator(episode_done_indices, obs, actions, logprobs, values, advantages, returns,
                                             rnn_hidden_states, rnn_cell_states, invalid_action_masks)
        for batch in data_generator:
            b_obs, b_actions, b_values, b_returns, b_logprobs, b_advantages, b_rnn_hidden_states, b_rnn_cell_states, b_loss_mask, b_invalid_action_masks = batch['vis_obs'], batch['actions'], \
                                                                                                                                                           batch['values'], batch['returns'], \
                                                                                                                                                           batch['log_probs'], batch['advantages'], \
                                                                                                                                                           batch["hxs"], batch["cxs"], batch["loss_mask"], \
                                                                                                                                                           batch["invalid_action_masks"]
            if args.norm_adv:
                b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

            new_values, _, _, newlogproba, entropy, _ = agent.get_action(
                b_obs,
                (b_rnn_hidden_states.unsqueeze(0), b_rnn_cell_states.unsqueeze(0)),
                seq_length=args.seq_length,
                action = b_actions.permute(1, 0).long(),
                invalid_action_masks = b_invalid_action_masks,
                envs = envs)
            ratio = (newlogproba - b_logprobs).exp()

            # Stats
            approx_kl = (b_logprobs - newlogproba).mean()

            # Policy loss
            pg_loss1 = -b_advantages * ratio
            pg_loss2 = -b_advantages * torch.clamp(ratio, 1-args.clip_coef, 1+args.clip_coef)
            pg_loss = torch.max(pg_loss1, pg_loss2)
            pg_loss = masked_mean(pg_loss, b_loss_mask)
            entropy_loss = masked_mean(entropy, b_loss_mask)

            # Value loss
            new_values = new_values.view(-1)
            if args.clip_vloss:
                v_loss_unclipped = ((new_values - b_returns) ** 2)
                v_clipped = b_values + torch.clamp(new_values - b_values, -args.clip_coef, args.clip_coef)
                v_loss_clipped = (v_clipped - b_returns)**2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * masked_mean(v_loss_max, b_loss_mask)
            else:
                v_loss = 0.5 *((new_values - b_returns) ** 2)

            loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
            optimizer.step()

        if args.kle_stop:
            if approx_kl > args.target_kl:
                break

    ## CRASH AND RESUME LOGIC:
    if args.prod_mode:
        if not os.path.exists(f"models/{experiment_name}"):
            os.makedirs(f"models/{experiment_name}")
            torch.save(agent.state_dict(), f"{wandb.run.dir}/agent.pt")
            wandb.save(f"agent.pt")
        else:
            if update % CHECKPOINT_FREQUENCY == 0:
                torch.save(agent.state_dict(), f"{wandb.run.dir}/agent.pt")

    # TRY NOT TO MODIFY: record rewards for plotting purposes
    writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]['lr'], global_step)
    writer.add_scalar("charts/update", update, global_step)
    writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
    writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
    writer.add_scalar("losses/entropy", entropy.mean().item(), global_step)
    writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
    if args.kle_stop or args.kle_rollback:
        writer.add_scalar("debug/pg_stop_iter", i_epoch_pi, global_step)
    writer.add_scalar("charts/sps", int(global_step / (time.time() - start_time)), global_step)
    print("SPS:", int(global_step / (time.time() - start_time)))

envs.close()
writer.close()
