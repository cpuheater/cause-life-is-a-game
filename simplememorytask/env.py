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



if __name__ == "__main__":
    env = SimpleMemoryTask()
    print(env.reset())

    for _ in range(1000):
        action = env.action_space.sample()
        observation, reward, done, done , info = env.step(action)
        print(f"observation {observation} reward: ${reward} ${done} ${info}")
    env.close()