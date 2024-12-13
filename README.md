# SIMPLE_SPREAD-MARL
The project involves experimentation with the SIMPLE SPREAD env the petting-zoo repository.

Details of the environment are present on their website

https://www.pettingzoo.ml/mpe/simple_spread

This GitHub repository contains two distinct approaches to training agents within the PettingZoo SIMPLE_SPREAD multi-agent environment:

1. **simple_spread_final**: This folder implements three variations of a Deep Q-Network (DQN) algorithm for agent training.  DQNs are model-free reinforcement learning algorithms that learn a Q-function to estimate the expected future reward for each action in a given state.  The three variations likely explore different architectures, hyperparameters, or training methodologies within the DQN framework.
2.  **SIMPLE_SPREAD WITH DDPG**: This folder utilizes Deep Deterministic Policy Gradient (DDPG), an actor-critic algorithm, for agent training.  DDPG is a model-free reinforcement learning algorithm suitable for continuous action spaces.  It learns both a policy (actor) to select actions and a value function (critic) to estimate the value of those actions. The actor-critic architecture allows for more stable and efficient learning compared to solely using a Q-function, particularly in continuous action spaces as found in many multi-agent environments.
