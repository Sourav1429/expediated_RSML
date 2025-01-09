import numpy as np

class ValueIteration:
    def __init__(self, num_states, num_actions, transition_probabilities, rewards, gamma=0.9, threshold=1e-6):
        """
        Initialize the ValueIteration class.

        :param num_states: Number of states
        :param num_actions: Number of actions
        :param transition_probabilities: Transition probability matrix (shape: [num_actions, num_states, num_states])
        :param rewards: Reward matrix (shape: [num_states, num_actions])
        :param gamma: Discount factor (default: 0.9)
        :param threshold: Convergence threshold for value iteration (default: 1e-6)
        """
        self.num_states = num_states
        self.num_actions = num_actions
        self.P = transition_probabilities
        self.R = rewards
        self.gamma = gamma
        self.threshold = threshold
        self.value_function = np.zeros(num_states)
        self.policy = np.zeros(num_states, dtype=int)

    def value_iteration(self):
        """
        Perform the Value Iteration algorithm to calculate the optimal value function and policy.
        """
        while True:
            delta = 0
            new_value_function = np.zeros(self.num_states)

            for s in range(self.num_states):
                action_values = np.zeros(self.num_actions)

                for a in range(self.num_actions):
                    action_values[a] = self.R[a, s] + self.gamma * np.sum(
                        self.P[a, s] * self.value_function
                    )

                # Update the value function for state s
                new_value_function[s] = np.max(action_values)
                delta = max(delta, abs(new_value_function[s] - self.value_function[s]))

            # Update the value function
            self.value_function = new_value_function

            # Check for convergence
            if delta < self.threshold:
                break

        # Derive the policy from the value function
        self.derive_policy()

    def derive_policy(self):
        """
        Derive the optimal policy from the calculated value function.
        """
        for s in range(self.num_states):
            action_values = np.zeros(self.num_actions)

            for a in range(self.num_actions):
                action_values[a] = self.R[a, s] + self.gamma * np.sum(
                    self.P[a, s] * self.value_function
                )

            # Choose the action with the highest value
            self.policy[s] = np.argmax(action_values)

    def get_value_function(self):
        """
        Get the calculated value function.

        :return: The value function (array of size [num_states])
        """
        return self.value_function

    def get_policy(self):
        """
        Get the derived optimal policy.

        :return: The optimal policy (array of size [num_states])
        """
        return self.policy
