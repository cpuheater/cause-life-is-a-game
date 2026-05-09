# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
from env import SimpleMemoryTask



@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "SimpleMemoryTask"
    """the id of the environment"""
    total_timesteps: int = 10000000
    """total timesteps of the experiments"""
    learning_rate: float = 2e-3
    """the learning rate of the optimizer"""
    num_envs: int = 12
    """the number of parallel game environments"""
    num_steps: int = 128
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""
    rnn_hidden_size=256
    """"""
    seq_length=64


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


def make_env(seed):
    def thunk():
        env = SimpleMemoryTask()
        env.action_space.seed(seed)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env
    return thunk



def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer
    
def recurrent_generator(episode_done_indices, obs, actions, logprobs, values, advantages, returns, rnn_hidden_states, cell_hidden_states):

    # Supply training samples
    samples = {
        'vis_obs': obs.permute(1, 0, 2).cpu().numpy(),
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
    num_sequences_per_batch = num_sequences // args.num_minibatches
    num_sequences_per_batch = [
                                  num_sequences_per_batch] * args.num_minibatches  # Arrange a list that determines the episode count for each mini batch
    remainder = num_sequences % args.num_minibatches
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


class Agent(nn.Module):
    def __init__(self, envs, rnn_input_size=256, rnn_hidden_size=256):
        super(Agent, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(3, args.rnn_hidden_size),
            nn.ReLU(True)
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

    def get_action_and_value(self, x, rnn_state, sequence_length=1, action=None):
        x, rnn_state = self.forward(x, rnn_state, sequence_length)
        value = self.critic(x)
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), value, rnn_state, 

    def get_value(self, x, rnn_state):
        x, _ = self.forward(x, rnn_state)
        return self.critic(x)        


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(seed=args.seed+i) for i in range(args.num_envs)],
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    agent = Agent(envs, rnn_hidden_size=args.rnn_hidden_size, rnn_input_size=args.rnn_hidden_size).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rnn_hidden_states = torch.zeros((args.num_steps, args.num_envs, args.rnn_hidden_size)).to(device)
    rnn_cell_states = torch.zeros((args.num_steps, args.num_envs, args.rnn_hidden_size)).to(device)


    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    rnn_hidden_state = torch.zeros((1, args.num_envs, args.rnn_hidden_size)).to(device)
    rnn_cell_state = torch.zeros((1, args.num_envs, args.rnn_hidden_size)).to(device)
    
    for iteration in range(1, args.num_iterations + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow
        episode_done_indices = [[] for w in range(args.num_envs)]
        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done
            rnn_hidden_states[step] = rnn_hidden_state
            rnn_cell_states[step] = rnn_cell_state

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value, (rnn_hidden_state, rnn_cell_state) = agent.get_action_and_value(next_obs, (rnn_hidden_state, rnn_cell_state))
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)
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

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs, (rnn_hidden_state, rnn_cell_state)).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # Optimizing the policy and value network
        clipfracs = []
        for epoch in range(args.update_epochs):
            data_generator = recurrent_generator(episode_done_indices, obs, actions, logprobs, values, advantages, returns,
                                                rnn_hidden_states, rnn_cell_states)            
            for batch in data_generator:
                b_obs, b_actions, b_values, b_returns, b_logprobs, b_advantages, b_rnn_hidden_states, b_rnn_cell_states, b_loss_mask = batch['vis_obs'], batch['actions'], \
                                                                                                                        batch['values'], batch['returns'], \
                                                                                                                        batch['log_probs'], batch['advantages'], \
                                                                                                                        batch["hxs"], batch["cxs"], batch["loss_mask"]            
                max_sequence_length = batch['max_sequence_length']

                _, newlogprob, entropy, newvalue, rnn_states = agent.get_action_and_value(b_obs, (b_rnn_hidden_states.unsqueeze(0), b_rnn_cell_states.unsqueeze(0)), 
                                                                              sequence_length=max_sequence_length, action=b_actions.long())
                logratio = newlogprob - b_logprobs
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2)
                pg_loss = masked_mean(pg_loss, b_loss_mask)
                entropy_loss = masked_mean(entropy, b_loss_mask)


                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns) ** 2
                    v_clipped = b_values + torch.clamp(
                        newvalue - b_values,
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * masked_mean(v_loss_max, b_loss_mask).mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns) ** 2).mean()

                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()
