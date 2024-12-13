# MARLDDPG
The SIMPLE_SPREAD WITH DDPG folder details two DDPG experiments:
**Independent Actor-Critic Networks**: Three independent pairs of actor-critic networks are used, one pair for each agent. Each agent learns its own policy and value function independently.  This allows for potentially more diverse strategies, but may hinder overall team performance.
**Parameter Sharing Actor-Critic Network**: A single pair of actor-critic networks is shared across all three agents. The parameters of both the actor and critic networks are shared, enforcing a common policy and value function across all agents.  This approach promotes collaboration and potentially leads to better coordinated behavior.
Both experiments aim to improve the reward function.  Note that because DDPG is designed for continuous action spaces, careful consideration should be given to the action space definition within the source code.  For detailed guidance on handling this, refer to:

https://github.com/PettingZoo-Team/PettingZoo/issues/249#issuecomment-728299094
