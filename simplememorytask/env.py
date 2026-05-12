from gymnasium import spaces
import numpy as np
import time
import os
from reprint import output
import gymnasium

class SimpleMemoryTask(gymnasium.Env):
    """
    """
    def __init__(self, step = 0.2):
        self._step_size = step
        self._lower = -1
        self._upper = 1
        self._all_positions = np.append(np.arange(self._lower, self._upper, self._step_size), self._upper).round(decimals=2)
        self._time_penalty = 0.1    
        self._num_show_steps = 0
        
        num_steps = int( 0.4 / self._step_size)
        lower = min(- 2.0 * self._step_size, -num_steps * self._step_size)
        upper = max( 3.0 * self._step_size, self._step_size, (num_steps + 1) * self._step_size)

        self.possible_positions = np.arange(lower, upper, self._step_size).clip(-1 + self._step_size, 1 - self._step_size).round(decimals=2)
        self.op = None
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(0, 1, shape=(3,))

    def reset(self, **kwargs):
        goals = np.asarray([-1.0, 1.0], dtype=np.float32)
        self._goals = goals[np.random.permutation(2)]
        goal_index = 0 if self._goals[0] == 1 else self._all_positions.shape[0] - 1
        self._position = np.random.choice(self.possible_positions)
        #print(f"initial position {self._position}")
        index = np.where(self._all_positions == self._position)
        self._min_steps = abs(index[0][0]-goal_index) - 1
        self._rewards = []
        self._step_count = 0                
        obs = np.asarray([self._goals[0], self._position, self._goals[1]], dtype=np.float32)
        return obs, {}

    def step(self, action):
        reward = 0.0
        done = False
        info = None
        if self._num_show_steps > self._step_count:
            self._position += self._step_size if action == 1 else -self._step_size 
            self._position = np.round(self._position, 2)

            obs = np.asarray([self._goals[0], self._position, self._goals[1]], dtype=np.float32)
        else:
            self._position += self._step_size if action == 1 else -self._step_size
            self._position = np.round(self._position, 2)
            obs = np.asarray([0.0, self._position, 0.0], dtype=np.float32)

        if self._position == -1.0:
            if self._goals[0] == 1.0:
                reward = 1.0 + self._min_steps * self._time_penalty
            else:
                reward = -1.0 
            done = True
        elif self._position == 1.0:
            if self._goals[1] == 1.0:
                reward = 1.0 + self._min_steps * self._time_penalty
            else:
                reward = -1.0 
            done = True
        else:
            reward -= self._time_penalty
        self._rewards.append(reward)

        if done:
            info = {"reward": round(sum(self._rewards)),
                    "length": len(self._rewards)}
        else:
            info = {}
        self._step_count += 1
        return obs, reward, done, done, info 

if __name__ == "__main__":
    env = SimpleMemoryTask()
    obs = env.reset()
    print(f"reset observation {obs}")
    for _ in range(1000):
        action = env.action_space.sample()
        action = 1 if env._goals[0] == 1 else 1
        observation, reward, terminated, truncated , info = env.step(action)
        done = np.logical_or(terminated, truncated)
        print(f"observation {observation} reward: ${reward} ${done} ${info}")
        if done:
            obs, info = env.reset()
            print(f"reset observation {obs} ${info}")    
    env.close()            