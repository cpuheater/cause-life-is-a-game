import gymnasium
import numpy as np
import random
from gymnasium import spaces
from typing import Optional


class SimpleMemoryTask(gymnasium.Env):
    def __init__(self):
        super(SimpleMemoryTask, self).__init__()
        self.action_space = spaces.Discrete(2)
        self._step_size = 0.2
        self._min_steps = int(1.0 / self._step_size) + 1
        self._time_penalty = 0.1
        self._num_show_steps = 2
        self.observation_space = spaces.Box(0, 1, shape=(3,))

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        self._rewards = []
        self._position = 0.0
        self._step_count = 0
        goals = np.asarray([-1.0, 1.0])
        self._goals = goals[np.random.permutation(2)]
        vec_obs = np.asarray([self._goals[0], self._position, self._goals[1]])
        return vec_obs, {}

    def step(self, action):
        self._position += self._step_size if action == 1 else -self._step_size
        if self._num_show_steps > self._step_count:
            vec_obs = np.asarray([self._goals[0], self._position, self._goals[1]])
        else:
            vec_obs = np.asarray([0.0, self._position, 0.0])
        reward = 0.0
        done = False
        if self._position == -1.0:
            if self._goals[0] == 1.0:
                reward += 1.0 + self._min_steps * self._time_penalty
            else:
                reward -= 1.0 + self._min_steps * self._time_penalty
            done = True
        elif self._position == 1.0:
            if self._goals[1] == 1.0:
                reward += 1.0 + self._min_steps * self._time_penalty
            else:
                reward -= 1.0 + self._min_steps * self._time_penalty
            done = True
        else:
            reward -= self._time_penalty
        self._rewards.append(reward)

        if done:
            info = {"reward": sum(self._rewards),
                    "length": len(self._rewards)}
        else:
            info = {}

        self._step_count += 1

        if done:
            vec_obs = self.reset()

        return vec_obs, reward, done, done, info

    def close(self):
        pass
    def render(self, mode='human'):
        pass


class PocMemoryEnv(gymnasium.Env):
    """
    Proof of Concept Memory Environment

    This environment is intended to assess whether the underlying recurrent policy is working or not.
    The environment is based on a one dimensional grid where the agent can move left or right.
    At both ends, a goal is spawned that is either punishing or rewarding.
    During the very first two steps, the agent gets to know which goal leads to a positive or negative reward.
    Afterwards, this information is hidden in the agent's observation.
    The last value of the agent's observation is its current position inside the environment.
    Optionally and to increase the difficulty of the task, the agent's position can be frozen until the goal information is hidden.
    To further challenge the agent, the step_size can be decreased.
    """
    def __init__(self, step_size:float=0.2, glob:bool=False, freeze:bool=False):
        """
        Args:
            step_size {float} -- Step size of the agent. Defaults to 0.2.
            glob {bool} -- Whether to sample starting positions across the entire space. Defaults to False.
            freeze_agent {bool} -- Whether to freeze the agent's position until goal positions are hidden. Defaults to False.
        """
        super(PocMemoryEnv, self).__init__()
        self.freeze = freeze
        self._step_size = step_size
        self._min_steps = int(1.0 / self._step_size) + 1
        self._time_penalty = 0.1
        self._num_show_steps = 2    # this should determine for how many steps the goal is visible

        # Create an array with possible positions
        # Valid local positions are one tick away from 0.0 or between -0.4 and 0.4
        # Valid global positions are between -1 + step_size and 1 - step_size
        # Clipping has to be applied because step_size is a variable now
        num_steps = int( 0.4 / self._step_size)
        lower = min(- 2.0 * self._step_size, -num_steps * self._step_size) if not glob else -1  + self._step_size
        upper = max( 3.0 * self._step_size, self._step_size, (num_steps + 1) * self._step_size) if not glob else 1

        self.possible_positions = np.arange(lower, upper, self._step_size).clip(-1 + self._step_size, 1 - self._step_size)
        self.possible_positions = list(map(lambda x: round(x, 2), self.possible_positions)) # fix floating point errors

        self.op = None


    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        """Resets the agent to a random start position and spawns the two possible goals randomly."""
        # Sample a random start position
        self._position = np.random.choice(self.possible_positions)
        self._rewards = []
        self._step_count = 0
        goals = np.asarray([-1.0, 1.0], dtype=np.float32)
        # Determine the goal
        self._goals = goals[np.random.permutation(2)]
        obs = np.asarray([self._goals[0], self._position, self._goals[1]], dtype=np.float32)
        return obs, {}

    @property
    def observation_space(self):
        """
        Returns:
            {spaces.Box}: The agent observes its current position and the goal locations, which are masked eventually.
        """
        return spaces.Box(low = 0, high = 1.0, shape = (3,), dtype = np.float32)

    @property
    def action_space(self):
        """
        Returns:
            {spaces.Discrete}: The agent has two actions: going left or going right
        """
        return spaces.Discrete(2)

    def step(self, action):
        """
        Executes the agents action in the environment if the agent is allowed to move.

        Arguments:
            action {list} -- The agent action which should be executed.

        Returns:
            {numpy.ndarray} -- Observation of the agent.
            {float} -- Reward for the agent.
            {bool} -- Done flag whether the episode has terminated.
            {dict} -- Information about episode reward, length and agents success reaching the goal position
        """
        reward = 0.0
        done = False
        info = None
        success = False
        action = action

        if self._num_show_steps > self._step_count:
            # Execute the agent action if agent is allowed to move
            self._position += self._step_size * (1 - self.freeze) if action == 1 else -self._step_size * (1 - self.freeze)
            self._position = np.round(self._position, 2)

            obs = np.asarray([self._goals[0], self._position, self._goals[1]], dtype=np.float32)

            if self.freeze: # Check if agent is allowed to move
                self._step_count += 1
                self._rewards.append(reward)
                return obs, reward, done, info

        else:
            self._position += self._step_size if action == 1 else -self._step_size
            self._position = np.round(self._position, 2)
            obs = np.asarray([0.0, self._position, 0.0], dtype=np.float32) # mask out goal information

        # Determine the reward function and episode termination
        if self._position == -1.0:
            if self._goals[0] == 1.0:
                reward += 1.0 + self._min_steps * self._time_penalty
                success = True
            else:
                reward -= 1.0 + self._min_steps * self._time_penalty
            done = True
        elif self._position == 1.0:
            if self._goals[1] == 1.0:
                reward += 1.0 + self._min_steps * self._time_penalty
                success = True
            else:
                reward -= 1.0 + self._min_steps * self._time_penalty
            done = True
        else:
            reward -= self._time_penalty
        self._rewards.append(reward)

        # Wrap up episode information
        if done:
            info = {"success": success,
                    "reward": sum(self._rewards),
                    "length": len(self._rewards)}
        else:
            info = {}

        # Increase step count
        self._step_count += 1

        return obs, reward, done, done, info

    def render(self):
        pass

    def close(self):
        """
        Clears the used resources properly.
        """
        if self.op is not None:
            self.op.clear()
            self.op = None


if __name__ == "__main__":
    env = SimpleMemoryTask()
    print(env.reset())

    for _ in range(1000):
        action = env.action_space.sample()
        observation, reward, done, done , info = env.step(action)
        print(f"observation {observation} reward: ${reward} ${done} ${info}")
    env.close()