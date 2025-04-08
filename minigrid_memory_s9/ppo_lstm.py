# https://github.com/facebookresearch/torchbeast/blob/master/torchbeast/core/environment.py

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

import argparse
from distutils.util import strtobool
import numpy as np
import time
import random
import os
import gymnasium as gym


class ObservationWrapper(gym.ObservationWrapper):

    def __init__(self, env):
        super().__init__(env)
        w, h, num_channels = env.observation_space["image"].shape
        new_shape = (num_channels, w, h)
        self.observation_space = gym.spaces.Box(0, 255, shape=new_shape, dtype=np.float32)

    def observation(self, observation):
        observation = observation['image'].astype('float32')
        return np.transpose(observation,(2,0,1))


def make_env(env_id, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return ObservationWrapper(env)
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

def masked_mean(tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Returns the mean of the tensor but ignores the values specified by the mask.
    This is used for masking out the padding of loss functions.

    Args:
        tensor {Tensor}: The to be masked tensor
        mask {Tensor}: The mask that is used to mask out padded values of a loss function

    Returns:
        {tensor}: Returns the mean of the masked tensor.
    """

    return (tensor.T * mask).sum() / torch.clamp((torch.ones_like(tensor.T) * mask).float().sum(), min=1.0)

class Agent(nn.Module):
    def __init__(self, envs, frames=3, rnn_input_size=256, rnn_hidden_size=256):
        super(Agent, self).__init__()
        self.network = nn.Sequential(
            #Scale(1/255),
            layer_init(nn.Conv2d(frames, 16, kernel_size=(1, 1), padding=0)),
            nn.LeakyReLU(),
            layer_init(nn.Conv2d(16, 20, kernel_size=(1, 1), padding=0)),
            nn.LeakyReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(980, rnn_input_size))
        )

        self.rnn = nn.LSTM(rnn_input_size, rnn_hidden_size, batch_first=True)
        for name, param in self.rnn.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                nn.init.orthogonal_(param, np.sqrt(2))
        self.actor = layer_init(nn.Linear(rnn_hidden_size, 3), std=0.01)
        self.critic = layer_init(nn.Linear(rnn_hidden_size, 1), std=1)

    def forward(self, x, rnn_state, sequence_length=1):
        x = self.network(x)
        if sequence_length == 1:
            x, rnn_state = self.rnn(x.unsqueeze(1), rnn_state)
            x = x.squeeze(1)
        else:
            x_shape = tuple(x.size())
            x = x.reshape((x_shape[0] // sequence_length), sequence_length, x_shape[1])
            x, rnn_state = self.rnn(x)
            x_shape = tuple(x.size())
            x = x.reshape(x_shape[0] * x_shape[1], x_shape[2])
        return x, rnn_state

    def get_action(self, x, rnn_state, sequence_length=1, action=None):
        x, rnn_state = self.forward(x, rnn_state, sequence_length)
        value = self.critic(x)
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return value, rnn_state, action, probs.log_prob(action), probs.entropy()

    def get_value(self, x, rnn_state):
        x, _ = self.forward(x, rnn_state)
        return self.critic(x)

def recurrent_generator(episode_done_indices, obs, actions, logprobs, values, advantages, returns, rnn_hidden_states, cell_hidden_states):

    # Supply training samples
    samples = {
        'vis_obs': obs.permute(1, 0, 2, 3, 4).cpu().numpy(),
        'actions': actions.permute(1, 0).cpu().numpy(),
        'values': values.permute(1, 0).cpu().numpy(),
        'log_probs': logprobs.permute(1, 0).cpu().numpy(),
        'advantages': advantages.permute(1, 0).cpu().numpy(),
        'returns': returns.permute(1, 0).cpu().numpy(),
        'loss_mask': np.ones((args.num_envs, args.num_steps), dtype=np.float32),
        "hxs": rnn_hidden_states.permute(1, 0, 2).cpu().numpy(),
        "cxs": cell_hidden_states.permute(1, 0, 2).cpu().numpy()
    }

    max_sequence_length = 1
    # Append the index of the last element of a trajectory as well, as it "artifically" marks the end of an episode
    for w in range(args.num_envs):
        if len(episode_done_indices[w]) == 0 or episode_done_indices[w][-1] != args.num_steps - 1:
            episode_done_indices[w].append(args.num_steps - 1)

    # Split vis_obs, vec_obs, values, advantages, actions and log_probs into episodes and then into sequences
    for key, value in samples.items():
        sequences = []
        for w in range(args.num_envs):
            start_index = 0
            for done_index in episode_done_indices[w]:
                # Split trajectory into episodes
                episode = value[w, start_index:done_index + 1]
                start_index = done_index + 1
                # Split episodes into sequences
                if args.seq_length > 0:
                    for start in range(0, len(episode), args.seq_length): # step min seq_length
                        end = start + args.seq_length
                        sequences.append(episode[start:end])
                    max_sequence_length = args.seq_length
                else:
                    # If the sequence length is not set to a proper value, sequences will be based on episodes
                    sequences.append(episode)
                    max_sequence_length = len(episode) if len(
                        episode) > max_sequence_length else max_sequence_length

        # Apply zero-padding to ensure that each episode has the same length
        # Therfore we can train batches of episodes in parallel instead of one episode at a time
        for i, sequence in enumerate(sequences):
            sequences[i] =  pad_sequence(sequence, max_sequence_length)

        # Stack episodes (target shape: (Episode, Step, Data ...) & apply data to the samples dict
        samples[key] = np.stack(sequences, axis=0)
        if (key == "hxs" or key == "cxs"):
            # Select the very first recurrent cell state of a sequence and add it to the samples
            samples[key] = samples[key][:, 0]

    # Store important information
    num_sequences = len(samples["values"])
    actual_sequence_length = max_sequence_length

    # Flatten all samples
    samples_flat = {}
    for key, value in samples.items():
        if (not key == "hxs" and not key == "cxs"):
            value = value.reshape(value.shape[0] * value.shape[1], *value.shape[2:])
        samples_flat[key] = torch.tensor(value, dtype=torch.float32, device=device)


    #generator
    num_sequences_per_batch = num_sequences // args.n_minibatch
    num_sequences_per_batch = [
                                  num_sequences_per_batch] * args.n_minibatch  # Arrange a list that determines the episode count for each mini batch
    remainder = num_sequences % args.n_minibatch
    for i in range(remainder):
        num_sequences_per_batch[i] += 1
    indices = np.arange(0, num_sequences * actual_sequence_length).reshape(num_sequences, actual_sequence_length)
    sequence_indices = torch.randperm(num_sequences)
    # At this point it is assumed that all of the available training data (values, observations, actions, ...) is padded.

    # Compose mini batches
    start = 0
    for num_sequences in num_sequences_per_batch:
        end = start + num_sequences
        mini_batch_indices = indices[sequence_indices[start:end]].reshape(-1)
        mini_batch = {}
        for key, value in samples_flat.items():
            if key != "hxs" and key != "cxs":
                mini_batch[key] = value[mini_batch_indices].to(device)
            else:
                # Collect recurrent cell states
                mini_batch[key] = value[sequence_indices[start:end]].to(device)
            mini_batch['max_sequence_length'] = max_sequence_length
        start = end
        yield mini_batch

def pad_sequence(sequence, target_length):
    # If a tensor is provided, convert it to a numpy array
    if isinstance(sequence, torch.Tensor):
        sequence = sequence.numpy()
    # Determine the number of zeros that have to be added to the sequence
    delta_length = target_length - len(sequence)
    # If the sequence is already as long as the target length, don't pad
    if delta_length <= 0:
        return sequence
    # Construct array of zeros
    if len(sequence.shape) > 1:
        # Case: pad multi-dimensional array like visual observation
        padding = np.zeros(((delta_length,) + sequence.shape[1:]), dtype=sequence.dtype)
        # padding = np.full(((delta_length,) + sequence.shape[1:]), sequence[0], dtype=sequence.dtype) # experimental
    else:
        padding = np.zeros(delta_length, dtype=sequence.dtype)
        # padding = np.full(delta_length, sequence[0], dtype=sequence.dtype) # experimental
    # Concatenate the zeros to the sequence
    return np.concatenate((sequence, padding), axis=0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PPO agent')
    # Common arguments
    parser.add_argument('--exp-name', type=str, default=os.path.basename(__file__).rstrip(".py"),
                        help='the name of this experiment')
    parser.add_argument('--env-id', type=str, default="MiniGrid-MemoryS9-v0",
                        help='the id of the gym environment')
    parser.add_argument('--learning-rate', type=float, default=1.0e-3,
                        help='the learning rate of the optimizer')
    parser.add_argument('--seed', type=int, default=1,
                        help='seed of the experiment')
    parser.add_argument('--total-timesteps', type=int, default=3000000,
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
    parser.add_argument('--rnn-hidden-size', type=int, default=256,
                        help='rnn hidden size')
    parser.add_argument('--seq-length', type=int, default=64,
                        help='seq length')

    # Algorithm specific arguments
    parser.add_argument('--n-minibatch', type=int, default=8,
                        help='the number of mini batch')
    parser.add_argument('--num-envs', type=int, default=8,
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
    #if not args.seed:
    args.seed = int(time.time())

    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.n_minibatch)
    start_time = time.time()
    experiment_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{time.strftime('%Y_%m_%d__%H_%M_%S')}"
    writer = SummaryWriter(f"runs/{experiment_name}")
    writer.add_text('hyperparameters', "|param|value|\n|-|-|\n%s" % ('\n'.join([f"|{key}|{value}|" for key, value in vars(args).items()])))
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

    #envs = VecPyTorch(DummyVecEnv([make_env(args.gym_id, args.seed+i, i) for i in range(args.num_envs)]), device)
    envs = gym.vector.AsyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, experiment_name) for i in range(args.num_envs)],
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    rnn_hidden_size = args.rnn_hidden_size
    agent = Agent(envs, rnn_hidden_size=args.rnn_hidden_size, rnn_input_size=args.rnn_hidden_size).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    if args.anneal_lr:
        # https://github.com/openai/baselines/blob/ea25b9e8b234e6ee1bca43083f8f3cf974143998/baselines/ppo2/defaults.py#L20
        lr = lambda f: f * args.learning_rate

    # ALGO Logic: Storage for epoch data
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rnn_hidden_states = torch.zeros((args.num_steps, args.num_envs, rnn_hidden_size)).to(device)
    rnn_cell_states = torch.zeros((args.num_steps, args.num_envs, rnn_hidden_size)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    # Note how `next_obs` and `next_done` are used; their usage is equivalent to
    # https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/84a7582477fb0d5c82ad6d850fe476829dddd2e1/a2c_ppo_acktr/storage.py#L60
    next_obs, _ = envs.reset()
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    rnn_hidden_state = torch.zeros((1, args.num_envs, rnn_hidden_size)).to(device)
    rnn_cell_state = torch.zeros((1, args.num_envs, rnn_hidden_size)).to(device)
    num_updates = args.total_timesteps // args.batch_size
    for update in range(1, num_updates+1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = lr(frac)
            optimizer.param_groups[0]['lr'] = lrnow
        episode_done_indices = [[] for w in range(args.num_envs)]
        # TRY NOT TO MODIFY: prepare the execution of the game.
        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs
            obs[step] = next_obs
            dones[step] = next_done
            rnn_hidden_states[step] = rnn_hidden_state
            rnn_cell_states[step] = rnn_cell_state

            # ALGO LOGIC: put action logic here
            with torch.no_grad():
                value, (rnn_hidden_state, rnn_cell_state), action, logproba, _ = agent.get_action(obs[step], (rnn_hidden_state, rnn_cell_state))

            values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logproba

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step], next_done, next_obs = torch.tensor(reward).to(device).view(-1), torch.Tensor(next_done).to(device), torch.Tensor(next_obs).to(device)
            mask = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in next_done]).to(device)
            rnn_hidden_state = rnn_hidden_state * mask
            rnn_cell_state = rnn_cell_state * mask
            indices = torch.nonzero(next_done).flatten().tolist()
            [episode_done_indices[index].append(step) for index in indices]

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

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
                                                rnn_hidden_states, rnn_cell_states)
            for batch in data_generator:
                b_obs, b_actions, b_values, b_returns, b_logprobs, b_advantages, b_rnn_hidden_states, b_rnn_cell_states, b_loss_mask = batch['vis_obs'], batch['actions'], \
                                                                                                                        batch['values'], batch['returns'], \
                                                                                                                        batch['log_probs'], batch['advantages'], \
                                                                                                                        batch["hxs"], batch["cxs"], batch["loss_mask"]
                max_sequence_length = batch['max_sequence_length']
                if args.norm_adv:
                    b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

                newvalues, rnn_states, _, newlogproba, entropy = agent.get_action(b_obs, (b_rnn_hidden_states.unsqueeze(0), b_rnn_cell_states.unsqueeze(0)),
                                                                        sequence_length=max_sequence_length, action=b_actions.long())
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
                new_values = newvalues.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = ((new_values - b_returns) ** 2)
                    v_clipped = b_values + torch.clamp(new_values - b_values, -args.clip_coef, args.clip_coef)
                    v_loss_clipped = (v_clipped - b_returns)**2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * masked_mean(v_loss_max, b_loss_mask)
                else:
                    v_loss = 0.5 * ((new_values - b_returns) ** 2).mean()

                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                grad_norm = sum(p.grad.detach().data.norm(2).item() ** 2 for p in agent.parameters()) ** 0.5
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()


        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/grad_norm", grad_norm, global_step)
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]['lr'], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy.mean().item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        if args.kle_stop or args.kle_rollback:
            writer.add_scalar("debug/pg_stop_iter", i_epoch_pi, global_step)
        writer.add_scalar("charts/sps", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()
