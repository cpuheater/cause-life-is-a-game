import cv2
import gymnasium
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import vizdoom.gymnasium_wrapper  # noqa
from vizdoom import gymnasium_wrapper  



DEFAULT_ENV = "VizdoomBasic-v0"
AVAILABLE_ENVS = [env for env in gymnasium.envs.registry.keys() if "Vizdoom" in env]
ENV = "VizdoomBasic-v0"
# Height and width of the resized image
IMAGE_SHAPE = (60, 80)
TB_LOG="./runs/sb_ppo_tensorboard/"

# Training parameters
TRAINING_TIMESTEPS = 10240
N_STEPS = 128
N_ENVS = 8
FRAME_SKIP = 4


class ObservationWrapper(gymnasium.ObservationWrapper):
    """
    ViZDoom environments return dictionaries as observations, containing
    the main image as well other info.
    The image is also too large for normal training.

    This wrapper replaces the dictionary observation space with a simple
    Box space (i.e., only the RGB image), and also resizes the image to a
    smaller size.

    NOTE: Ideally, you should set the image size to smaller in the scenario files
          for faster running of ViZDoom. This can really impact performance,
          and this code is pretty slow because of this!
    """

    def __init__(self, env, shape=IMAGE_SHAPE):
        super().__init__(env)
        self.image_shape = shape
        self.image_shape_reverse = shape[::-1]
        self.env.frame_skip = FRAME_SKIP

        # Create new observation space with the new shape
        print(env.observation_space)
        num_channels = env.observation_space["screen"].shape[-1]
        new_shape = (shape[0], shape[1], num_channels)
        self.observation_space = gymnasium.spaces.Box(
            0, 255, shape=new_shape, dtype=np.uint8
        )

    def observation(self, observation):
        observation = cv2.resize(observation["screen"], self.image_shape_reverse)
        return observation

def main():
    # Create multiple environments: this speeds up training with PPO
    # We apply two wrappers on the environment:
    #  1) The above wrapper that modifies the observations (takes only the image and resizes it)
    #  2) A reward scaling wrapper. Normally the scenarios use large magnitudes for rewards (e.g., 100, -100).
    #     This may lead to unstable learning, and we scale the rewards by 1/100
    def wrap_env(env):
        env = ObservationWrapper(env)
        env = gymnasium.wrappers.TransformReward(env, lambda r: r * 0.01)
        return env

    envs = make_vec_env(ENV, n_envs=N_ENVS, wrapper_class=wrap_env)

    agent = PPO("CnnPolicy", envs, n_steps=N_STEPS, verbose=1, device="cuda", tensorboard_log=TB_LOG)
    
    # Do the actual learning
    # This will print out the results in the console.
    # If agent gets better, "ep_rew_mean" should increase steadily
    agent.learn(total_timesteps=TRAINING_TIMESTEPS)

    agent.save("sb_ppo")

if __name__ == "__main__":
    main()