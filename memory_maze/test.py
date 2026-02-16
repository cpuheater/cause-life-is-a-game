import gym
import os

# Set this if you are getting "Unable to load EGL library" error:
os.environ['MUJOCO_GL'] = 'glfw'  

env = gym.make('memory_maze:MemoryMaze-9x9-v0')
env = gym.make('memory_maze:MemoryMaze-11x11-v0')
env = gym.make('memory_maze:MemoryMaze-13x13-v0')
env = gym.make('memory_maze:MemoryMaze-15x15-v0')