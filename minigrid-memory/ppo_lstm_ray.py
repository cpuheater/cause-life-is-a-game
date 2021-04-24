import ray
import ray.rllib.agents.ppo as ppo
ray.init()
from ray.tune import register_env
import gym_minigrid
import gym
import numpy as np
from ray.rllib.utils.numpy import one_hot
from collections import deque

class OneHotWrapper(gym.core.ObservationWrapper):
    def __init__(self, env, vector_index, framestack):
        super().__init__(env)
        self.framestack = framestack
        # 49=7x7 field of vision; 11=object types; 6=colors; 3=state types.
        # +4: Direction.
        self.single_frame_dim = 49 * (11 + 6 + 3) + 4
        self.init_x = None
        self.init_y = None
        self.x_positions = []
        self.y_positions = []
        self.x_y_delta_buffer = deque(maxlen=100)
        self.vector_index = vector_index
        self.frame_buffer = deque(maxlen=self.framestack)
        for _ in range(self.framestack):
            self.frame_buffer.append(np.zeros((self.single_frame_dim, )))

        self.observation_space = gym.spaces.Box(
            0.0,
            1.0,
            shape=(self.single_frame_dim * self.framestack, ),
            dtype=np.float32)

    def observation(self, obs):
        # Debug output: max-x/y positions to watch exploration progress.
        if self.step_count == 0:
            for _ in range(self.framestack):
                self.frame_buffer.append(np.zeros((self.single_frame_dim, )))
            if self.vector_index == 0:
                if self.x_positions:
                    max_diff = max(
                        np.sqrt((np.array(self.x_positions) - self.init_x)**2 +
                                (np.array(self.y_positions) - self.init_y)**2))
                    self.x_y_delta_buffer.append(max_diff)
                    print("100-average dist travelled={}".format(
                        np.mean(self.x_y_delta_buffer)))
                    self.x_positions = []
                    self.y_positions = []
                self.init_x = self.agent_pos[0]
                self.init_y = self.agent_pos[1]

        # Are we carrying the key?
        # if self.carrying is not None:
        #    print("Carrying KEY!!")

        self.x_positions.append(self.agent_pos[0])
        self.y_positions.append(self.agent_pos[1])

        # One-hot the last dim into 11, 6, 3 one-hot vectors, then flatten.
        objects = one_hot(obs[:, :, 0], depth=11)
        colors = one_hot(obs[:, :, 1], depth=6)
        states = one_hot(obs[:, :, 2], depth=3)
        # Is the door we see open?
        # for x in range(7):
        #    for y in range(7):
        #        if objects[x, y, 4] == 1.0 and states[x, y, 0] == 1.0:
        #            print("Door OPEN!!")

        all_ = np.concatenate([objects, colors, states], -1)
        all_flat = np.reshape(all_, (-1, ))
        direction = one_hot(
            np.array(self.agent_dir), depth=4).astype(np.float32)
        single_frame = np.concatenate([all_flat, direction])
        self.frame_buffer.append(single_frame)
        return np.concatenate(self.frame_buffer)


def env_maker(config):
    name = config.get("name", "MiniGrid-Empty-5x5-v0")
    framestack = config.get("framestack", 4)
    env = gym.make(name)
    # Only use image portion of observation (discard goal and direction).
    env = gym_minigrid.wrappers.ImgObsWrapper(env)
    env = OneHotWrapper(
        env,
        config.vector_index if hasattr(config, "vector_index") else 0,
        framestack=framestack)
    return env

register_env("mini-grid", env_maker)

config = ppo.DEFAULT_CONFIG.copy()
config["env"] = "mini-grid"
config["env_config"] = {
    # Also works with:
    # - MiniGrid-MultiRoom-N4-S5-v0
    # - MiniGrid-MultiRoom-N2-S4-v0
    "name": "MiniGrid-Empty-8x8-v0",
    "framestack": 1,  # seems to work even w/o framestacking
}
#config["horizon"] = 15  # Make it impossible to reach goal by chance.
config["num_envs_per_worker"] = 4
config["model"]["fcnet_hiddens"] = [256, 256]
config["model"]["fcnet_activation"] = "relu"
config["num_sgd_iter"] = 8
config["num_workers"] = 0
config["framework"] = "torch"


min_reward = 0.001
stop = {
    "training_iteration": 25,
    "episode_reward_mean": min_reward,
}

agent = ppo.PPOTrainer(config)
for n in range(100):
    result = agent.train()
    print(f'episode_reward_mean: {result["episode_reward_mean"]}')