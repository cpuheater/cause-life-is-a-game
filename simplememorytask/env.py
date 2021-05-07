import gym
import numpy as np
import random
from gym import error, spaces

class SimpleMemoryTask:
    def __init__(self):
        self._action_space = spaces.Discrete(2)
        self._step_size = 0.2
        self._min_steps = int(1.0 / self._step_size) + 1
        self._time_penalty = 0.1
        self._num_show_steps = 2    # this should determine for how long the goal is visible
        self._vector_observation_space = (3,)

    @property
    def vector_observation_space(self):
        return self._vector_observation_space

    @property
    def visual_observation_space(self):
        return None

    @property
    def action_space(self):
        return self._action_space

    def reset(self):
        self._rewards = []
        self._position = 0.0
        self._step_count = 0
        goals = np.asarray([-1.0, 1.0])
        self._goals = goals[np.random.permutation(2)]
        vec_obs = np.asarray([self._goals[0], self._position, self._goals[1]])
        return None, vec_obs

    def step(self, action):
        # Execute action
        self._position += self._step_size if action == 1 else -self._step_size
        # Create vector observation
        if self._num_show_steps > self._step_count:
            vec_obs = np.asarray([self._goals[0], self._position, self._goals[1]])
        else:
            vec_obs = np.asarray([0.0, self._position, 0.0]) # mask out goal information

        # Determine reward and episode termination
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

        # Wrap up episode information
        if done:
            info = {"reward": sum(self._rewards),
                    "length": len(self._rewards)}
        else:
            info = None

        # Increase step count
        self._step_count += 1

        if done:
            vec_obs = self.reset()

        return vec_obs, reward, done, info

    def close(self):
        pass



env = SimpleMemoryTask()
print(env.reset())

for _ in range(1000):
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action) # take a random action
    print(f"observation {observation} reward: ${reward} ${done} ${info}")
env.close()