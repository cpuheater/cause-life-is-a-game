import numpy as np
import itertools
from random import sample, randint, random
import numpy as np
from vizdoom import DoomGame, Mode, ScreenFormat, ScreenResolution
import torch
#from skimage.transform import resize
import cv2
from collections import deque

class ReplayBuffer(object):
    def __init__(self, size):
        self._storage = []
        self._maxsize = size
        self._next_idx = 0

    def put(self, data):
        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def sample(self, batch_size):
        idxes = np.random.choice(len(self._storage), batch_size, replace=True)
        obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, action, reward, obs_tp1, done = data
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            dones.append(done)
        return np.array(obses_t), np.array(actions), np.array(rewards), np.array(obses_tp1), np.array(dones)


def initialize_vizdoom(config):
    game = DoomGame()
    game.load_config(config)
    #game.set_window_visible(False)
    game.set_mode(Mode.PLAYER)
    game.set_screen_format(ScreenFormat.GRAY8)
    game.set_screen_resolution(ScreenResolution.RES_640X480)
    game.init()
    return game

def preprocess(img, resolution):
    return torch.from_numpy(cv2.resize(img, resolution).astype(np.float32))

def stack_frames(stacked_frames, state, is_new_episode, resolution):
    frame = preprocess(state, resolution)

    if is_new_episode:
        stacked_frames = deque([np.zeros(resolution, dtype=np.int) for i in range(len(stacked_frames))], maxlen=4)
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_state = np.stack(stacked_frames, axis=2).transpose(2, 0, 1)
    else:
        stacked_frames.append(frame)
        stacked_state = np.stack(stacked_frames, axis=2).transpose(2, 0, 1)

    return stacked_state, stacked_frames