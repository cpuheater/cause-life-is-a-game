### Solving video games with Deep Reinforcement Learning!
Each file is a self-contained rl algorithm, tuned to solve that particular game. 
An algorithm might have one or many extensions. You can determine the name of the algorithm and the type of the extension from the name of the file.      
  
### List of algorithms:
* ppo  - Proximal Policy Optimization Algorithm (https://arxiv.org/abs/1707.06347)    
* ppo_lstm - ppo with recurrent policy using LSTM
* ppo_gru - ppo with recurrent policy using GRU 
* sac_dis - Soft Actor-Critic for Discrete Action Settings (https://arxiv.org/abs/1910.07207)
* a2c - Advantage Actor Critic 
* dqn - (https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf)
* sac - Soft Actor-Critic (https://arxiv.org/abs/1801.01290)
* ddqn - Double DQN (https://arxiv.org/abs/1509.06461)

### List of extensions:
* frame_stacking - stacking four consecutive frames
* vt - vision transformer as an encoder
* n_step - using n step returns
* relational - Relational Deep Reinforcement Learning (https://arxiv.org/abs/1806.01830)
* sil - Self imitation learning (https://arxiv.org/abs/1806.05635)  
* icm - 


The core algorithms are based on [cleanrl](https://github.com/vwxyzjn/cleanrl) and comes with many 
cleanrl goodies like: tensorboard logging, videos of gameplay capturing, experiment 
management with [weights and biases](https://wandb.ai/site). 
