### Solving video games with Deep Reinforcement Learning!
Each file is a self-contained rl algorithm, tuned to solve that particular game.
An algorithm might have one or many extensions. You can determine the name of the algorithm and the type of the extension from the name of the file.
The core algorithms are based on [cleanrl](https://github.com/vwxyzjn/cleanrl) and comes with many cleanrl goodies like: tensorboard logging, videos of gameplay capturing,
experiment management with [weights and biases](https://wandb.ai/site).


### List of algorithms:
* ppo  - Proximal Policy Optimization (https://arxiv.org/abs/1707.06347)
* ppo_lstm - PPO with recurrent policy using LSTM
* ppo_gru - PPO with recurrent policy using GRU
* sac_dis - Soft Actor-Critic for discrete action settings (https://arxiv.org/abs/1910.07207)
* a2c - Advantage Actor Critic
* dqn - Deep Q-Network (https://arxiv.org/abs/1509.06461)
* sac - Soft Actor-Critic (https://arxiv.org/abs/1801.01290)
* ddqn - Double DQN (https://arxiv.org/abs/1509.06461)
* dueling_dqn - Dueling DQN (https://arxiv.org/abs/1511.06581)
* ppo_slstm - xLSTM: Extended Long Short-Term Memory (https://arxiv.org/abs/2405.04517)

### List of extensions:
* ppo_separate - separate network for the actor and the critic
* frame_stacking - stacking four consecutive frames
* vt - vision transformer as an encoder
* n_step - using n step returns, Asynchronous Methods for Deep Reinforcement Learning (https://arxiv.org/pdf/1602.01783)
* relational - relational deep reinforcement learning (https://arxiv.org/abs/1806.01830)
* sil - self imitation learning (https://arxiv.org/abs/1806.05635)
* icm - curiosity driven exploration (https://arxiv.org/pdf/1705.05363.pdf)
* branching - action branching (https://arxiv.org/pdf/1711.08946.pdf)

