import numpy as np
import torch
import random
#from baselines.common.segment_tree import SumSegmentTree, MinSegmentTree
from torch.distributions.categorical import Categorical

class SegmentTree(object):
    def __init__(self, capacity, operation, neutral_element):
        """Build a Segment Tree data structure.

        https://en.wikipedia.org/wiki/Segment_tree

        Can be used as regular array, but with two
        important differences:

            a) setting item's value is slightly slower.
               It is O(lg capacity) instead of O(1).
            b) user has access to an efficient `reduce`
               operation which reduces `operation` over
               a contiguous subsequence of items in the
               array.

        Paramters
        ---------
        capacity: int
            Total size of the array - must be a power of two.
        operation: lambda obj, obj -> obj
            and operation for combining elements (eg. sum, max)
            must for a mathematical group together with the set of
            possible values for array elements.
        neutral_element: obj
            neutral element for the operation above. eg. float('-inf')
            for max and 0 for sum.
        """
        assert capacity > 0 and capacity & (capacity - 1) == 0, "capacity must be positive and a power of 2."
        self._capacity = capacity
        self._value = [neutral_element for _ in range(2 * capacity)]
        self._operation = operation

    def _reduce_helper(self, start, end, node, node_start, node_end):
        if start == node_start and end == node_end:
            return self._value[node]
        mid = (node_start + node_end) // 2
        if end <= mid:
            return self._reduce_helper(start, end, 2 * node, node_start, mid)
        else:
            if mid + 1 <= start:
                return self._reduce_helper(start, end, 2 * node + 1, mid + 1, node_end)
            else:
                return self._operation(
                    self._reduce_helper(start, mid, 2 * node, node_start, mid),
                    self._reduce_helper(mid + 1, end, 2 * node + 1, mid + 1, node_end)
                )

    def reduce(self, start=0, end=None):
        """Returns result of applying `self.operation`
        to a contiguous subsequence of the array.

            self.operation(arr[start], operation(arr[start+1], operation(... arr[end])))

        Parameters
        ----------
        start: int
            beginning of the subsequence
        end: int
            end of the subsequences

        Returns
        -------
        reduced: obj
            result of reducing self.operation over the specified range of array elements.
        """
        if end is None:
            end = self._capacity
        if end < 0:
            end += self._capacity
        end -= 1
        return self._reduce_helper(start, end, 1, 0, self._capacity - 1)

    def __setitem__(self, idx, val):
        # index of the leaf
        idx += self._capacity
        self._value[idx] = val
        idx //= 2
        while idx >= 1:
            self._value[idx] = self._operation(
                self._value[2 * idx],
                self._value[2 * idx + 1]
            )
            idx //= 2

    def __getitem__(self, idx):
        assert 0 <= idx < self._capacity
        return self._value[self._capacity + idx]

class SumSegmentTree(SegmentTree):
    def __init__(self, capacity):
        super(SumSegmentTree, self).__init__(
            capacity=capacity,
            operation=np.add,
            neutral_element=0.0
        )

    def sum(self, start=0, end=None):
        """Returns arr[start] + ... + arr[end]"""
        return super(SumSegmentTree, self).reduce(start, end)

    def find_prefixsum_idx(self, prefixsum):
        """Find the highest index `i` in the array such that
            sum(arr[0] + arr[1] + ... + arr[i - i]) <= prefixsum

        if array values are probabilities, this function
        allows to sample indexes according to the discrete
        probability efficiently.

        Parameters
        ----------
        perfixsum: float
            upperbound on the sum of array prefix

        Returns
        -------
        idx: int
            highest index satisfying the prefixsum constraint
        """
        assert 0 <= prefixsum <= self.sum() + 1e-5
        idx = 1
        while idx < self._capacity:  # while non-leaf
            if self._value[2 * idx] > prefixsum:
                idx = 2 * idx
            else:
                prefixsum -= self._value[2 * idx]
                idx = 2 * idx + 1
        return idx - self._capacity


class MinSegmentTree(SegmentTree):
    def __init__(self, capacity):
        super(MinSegmentTree, self).__init__(
            capacity=capacity,
            operation=min,
            neutral_element=float('inf')
        )

    def min(self, start=0, end=None):
        """Returns min(arr[start], ...,  arr[end])"""

        return super(MinSegmentTree, self).reduce(start, end)



# replay buffer...
class ReplayBuffer:
    def __init__(self, size):
        self._storage = []
        self._maxsize = size
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def add(self, obs_t, action, R, invalid_action_mask):
        data = (obs_t, action, R, invalid_action_mask)
        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def sample(self, batch_size):
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes)

    def _encode_sample(self, idxes):
        obses_t, actions, returns, invalid_action_masks = [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, action, R, invalid_action_mask = data
            obses_t.append(obs_t)
            actions.append(action)
            returns.append(R)
            invalid_action_masks.append(invalid_action_mask)
        return np.array(obses_t), np.array(actions), np.array(returns), np.array(invalid_action_masks)


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, size, alpha):
        super(PrioritizedReplayBuffer, self).__init__(size)
        assert alpha > 0
        self._alpha = alpha
        it_capacity = 1
        while it_capacity < size:
            it_capacity *= 2
        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0

    def add(self, *args, **kwargs):
        idx = self._next_idx
        super().add(*args, **kwargs)
        self._it_sum[idx] = self._max_priority ** self._alpha
        self._it_min[idx] = self._max_priority ** self._alpha

    def _sample_proportional(self, batch_size):
        res = []
        for _ in range(batch_size):
            mass = random.random() * self._it_sum.sum(0, len(self._storage) - 1)
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return res

    def update_priorities(self, idxes, priorities):
        assert len(idxes) == len(priorities)
        for idx, priority in zip(idxes, priorities):
            priority = max(priority, 1e-6)
            assert priority > 0
            assert 0 <= idx < len(self._storage)
            self._it_sum[idx] = priority ** self._alpha
            self._it_min[idx] = priority ** self._alpha
            self._max_priority = max(self._max_priority, priority)

    def sample(self, batch_size, beta):
        idxes = self._sample_proportional(batch_size)
        if beta > 0:
            weights = []
            p_min = self._it_min.min() / self._it_sum.sum()
            max_weight = (p_min * len(self._storage)) ** (-beta)

            for idx in idxes:
                p_sample = self._it_sum[idx] / self._it_sum.sum()
                weight = (p_sample * len(self._storage)) ** (-beta)
                weights.append(weight / max_weight)
            weights = np.array(weights)
        else:
            weights = np.ones_like(idxes, dtype=np.float32)
        encoded_sample = self._encode_sample(idxes)
        return tuple(list(encoded_sample) + [weights, idxes])


# self-imitation learning
class sil_module:
    def __init__(self, network, args, optimizer, envs):
        self.args = args
        self.network = network
        self.running_episodes = [[] for _ in range(self.args.num_envs)]
        self.optimizer = optimizer
        self.buffer = PrioritizedReplayBuffer(self.args.capacity, 0.6)
        # some other parameters...
        self.total_steps = []
        self.total_rewards = []
        self.envs = envs


    # add the batch information into it...
    def step(self, obs, actions, rewards, dones, invalid_action_masks, wins):
        for n in range(self.args.num_envs):
            self.running_episodes[n].append([obs[n], actions[n], rewards[n], invalid_action_masks[n], wins[n]])
        # to see if can update the episode...
        for n, done in enumerate(dones):
            if done:
                self.update_buffer(self.running_episodes[n])
                self.running_episodes[n] = []

    # train the sil model...
    def train_sil_model(self):
        for n in range(self.args.num_update):
            obs, actions, returns, invalid_action_masks, weights, idxes = self.sample_batch(self.args.batch_size)
            mean_adv, num_valid_samples = 0, 0
            if obs is not None:
                # need to get the masks
                # get basic information of network..
                obs = torch.tensor(obs, dtype=torch.float32)
                actions = torch.tensor(actions, dtype=torch.float32)
                returns = torch.tensor(returns, dtype=torch.float32).unsqueeze(1)
                weights = torch.tensor(weights, dtype=torch.float32).unsqueeze(1)
                invalid_action_masks = torch.tensor(invalid_action_masks, dtype=torch.float32)
                max_nlogp = torch.tensor(np.ones((len(idxes), 1)) * self.args.max_nlogp, dtype=torch.float32)
                if self.args.cuda:
                    obs = obs.cuda()
                    actions = actions.cuda()
                    returns = returns.cuda()
                    weights = weights.cuda()
                    max_nlogp = max_nlogp.cuda()
                    invalid_action_masks = invalid_action_masks.cuda()
                # start to next...
                value, _, action_log_probs, dist_entropy, _ = self.network.get_action(obs, actions,
                                                                                   invalid_action_masks, self.envs)
                #value, pi = self.model.get_action(obs, actions)
                #action_log_probs, dist_entropy = evaluate_actions_sil(pi, actions)
                action_log_probs = -action_log_probs
                clipped_nlogp = torch.min(action_log_probs, max_nlogp)
                # process returns
                advantages = returns - value
                advantages = advantages.detach()
                masks = (advantages.cpu().numpy() > 0).astype(np.float32)
                # get the num of vaild samples
                num_valid_samples = np.sum(masks)
                num_samples = np.max([num_valid_samples, self.args.mini_batch_size])
                # process the mask
                masks = torch.tensor(masks, dtype=torch.float32)
                if self.args.cuda:
                    masks = masks.cuda()
                # clip the advantages...
                clipped_advantages = torch.clamp(advantages, 0, self.args.clip)
                mean_adv = torch.sum(clipped_advantages) / num_samples
                mean_adv = mean_adv.item()
                # start to get the action loss...
                action_loss = torch.sum(clipped_advantages * weights * clipped_nlogp) / num_samples
                entropy_reg = torch.sum(weights * dist_entropy * masks) / num_samples
                policy_loss = action_loss - entropy_reg * self.args.sil_entropy_coef
                # start to process the value loss..
                # get the value loss
                delta = torch.clamp(value - returns, -self.args.clip, 0) * masks
                delta = delta.detach()
                value_loss = torch.sum(weights * value * delta) / num_samples
                total_loss = policy_loss + 0.5 * self.args.w_value * value_loss
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.args.max_grad_norm)
                self.optimizer.step()
                # update the priorities
                self.buffer.update_priorities(idxes, clipped_advantages.squeeze(1).cpu().numpy())
        return mean_adv, num_valid_samples

    # update buffer
    def update_buffer(self, trajectory):
        positive_reward = False
        for (ob, a, r, invalid_action_mask, win) in trajectory:
            if win > 0:
                positive_reward = True
                break
        if positive_reward:
            self.add_episode(trajectory)
            self.total_steps.append(len(trajectory))
            self.total_rewards.append(np.sum([x[2] for x in trajectory]))
            while np.sum(self.args.num_steps) > self.args.capacity and len(self.args.num_steps) > 1:
                self.total_steps.pop(0)
                self.total_rewards.pop(0)

    def add_episode(self, trajectory):
        obs = []
        actions = []
        rewards = []
        dones = []
        invalid_action_masks = []
        for (ob, action, reward, invalid_action_mask, _) in trajectory:
            if ob is not None:
                obs.append(ob)
            else:
                obs.append(None)
            actions.append(action)
            rewards.append(np.sign(reward))
            dones.append(False)
            invalid_action_masks.append(invalid_action_mask)
        dones[len(dones) - 1] = True
        returns = self.discount_with_dones(rewards, dones, self.args.gamma)
        for (ob, action, R, invalid_action_mask) in list(zip(obs, actions, returns, invalid_action_masks)):
            self.buffer.add(ob, action, R, invalid_action_mask)

    def fn_reward(self, reward):
        return np.sign(reward)

    def get_best_reward(self):
        if len(self.total_rewards) > 0:
            return np.max(self.total_rewards)
        return 0

    def num_episodes(self):
        return len(self.total_rewards)

    def num_steps(self):
        return len(self.buffer)

    def sample_batch(self, batch_size):
        if len(self.buffer) > 100:
            print(f"buffer size {len(self.buffer)}")
            batch_size = min(batch_size, len(self.buffer))
            return self.buffer.sample(batch_size, beta=self.args.sil_beta)
        else:
            return None, None, None, None, None, None

    def discount_with_dones(self, rewards, dones, gamma):
        discounted = []
        r = 0
        for reward, done in zip(rewards[::-1], dones[::-1]):
            r = reward + gamma * r * (1. - done)
            discounted.append(r)
        return discounted[::-1]

def evaluate_actions_sil(pi, actions):
    cate_dist = Categorical(pi)
    return cate_dist.log_prob(actions.squeeze(-1)).unsqueeze(-1), cate_dist.entropy().unsqueeze(-1)
