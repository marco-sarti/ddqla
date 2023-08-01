# Agent Class
The Agent class is an abstract class that implements an agent for reinforcement learning. This class can be extended to create specific agents for different environments.

## Constructor
The constructor of the Agent class accepts the following parameters:

- **num_actions**: The number of possible actions that the agent can take.
- **environment**: The environment in which the agent operates, represented as a numpy array.
- **fit_each_n_steps**: The number of steps after which the agent trains its models.
- **exploration_rate**: The probability that the agent chooses a random action instead of the optimal action.
- **exploration_rate_decay**: The decay rate of the exploration rate.
- **gamma**: The discount factor used in the Q-value update equation.
- **cumulative_rewards_max_length**: The maximum length of the array that keeps track of cumulative rewards.
- **memory_max_length**: The maximum length of the agent's memory.
- **memory_batch_size**: The batch size used for training the models.
- **allow_episode_tracking**: A boolean flag indicating whether the agent should track episodes.

## Methods
The Agent class implements the following methods:

- **start_episode**: Starts a new episode.
- **stop_episode**: Ends the current episode.
- **get_episodes**: Returns all recorded episodes.
- **reset_episodes**: Resets all recorded episodes.
- **is_memory_ready**: Checks whether the agent's memory is ready for training.
- **step**: Performs a step of the agent, choosing an action, receiving a reward, and updating the models.
- **get_last_cumulative_rewards**: Returns the sum of the last cumulative rewards.

In addition, the Agent class requires the following functions to be implemented in subclasses:

- **reset_state**: Resets the agent's state at the start of a new episode.
- **_get_reward**: Calculates the reward received by the agent for undertaking an action in a given state.
- **_get_model**: Returns the model used by the agent to learn the Q-value function.

## Usage Example
To use the Agent class, you need to extend it and implement the abstract methods. Here's an example of how this might be done:

    class MyAgent(Agent):
        def reset_state(self):
            # Implementation of state reset
    
        def _get_reward(self, action, environment):
            # Implementation of reward calculation
    
        def _get_model(self, state_features):
            # Implementation of the model

Once the subclass is defined, you can create an instance of the agent and use it as follows:

    agent = MyAgent(num_actions=4, environment=np.array([0, 0, 0, 0]))
    agent.start_episode()
    for _ in range(100):
        agent.step()
    agent.stop_episode()

## Notes
The Agent class is designed to be used with discrete environments and deep learning models. If you wish to use a continuous environment or a different learning model, you may need to make some modifications to the class.
