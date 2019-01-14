import random

class Agent():
    def __init__(self, env, epsilon=1.0, alpha=0.5, gamma=0.9, num_episodes_to_train=30000):
        self.env = env

        # Looks like n is number of valid actions from the souce code
        self.valid_actions = list(range(self.env.action_space.n))

        # Set parameters of the learning agent
        self.Q = dict()          # Q-table which will be a dictionary of tuples
        self.epsilon = epsilon   # Random exploration factor
        self.alpha = alpha       # Learning factor
        self.gamma = gamma       # Discount factor- closer to 1 learns well into distant future

        # epsilon will reduce linearly until it reaches 0 based on num_episodes_to_train
        # epsilon drops to 90% of its inital value in the first 30% of num_episodes_to_train
        # epsilon then drops to 10% of its initial value in the next 40% of num_episodes_to_train
        # epsilon finally becomes 0 in the final 30% of num_episodes_to_train
        self.num_episodes_to_train = num_episodes_to_train # Change epsilon each episode based on this
        self.small_decrement = (0.1 * epsilon) / (0.3 * num_episodes_to_train) # reduces epsilon slowly
        self.big_decrement = (0.8 * epsilon) / (0.4 * num_episodes_to_train) # reduces epilon faster

        self.num_episodes_to_train_left = num_episodes_to_train

    def update_parameters(self):
        """
        Update epsilon and alpha after each action
        Set them to 0 if not learning
        """
        if self.num_episodes_to_train_left > 0.7 * self.num_episodes_to_train:
            self.epsilon -= self.small_decrement
        elif self.num_episodes_to_train_left > 0.3 * self.num_episodes_to_train:
            self.epsilon -= self.big_decrement
        elif self.num_episodes_to_train_left > 0:
            self.epsilon -= self.small_decrement
        else:
            self.epsilon = 0.0
            self.alpha = 0.0

        self.num_episodes_to_train_left -= 1

    def create_Q_if_new_observation(self, observation):
        """
        Set intial Q values to 0.0 if observation not already in Q table
        """
        if observation not in self.Q:
            self.Q[observation] = dict((action, 0.0) for action in self.valid_actions)
            # print("obsv q:", self.Q[observation])
            # print("Q:", self.Q)
            # print("\n\n\n\n")
            # sys.exit(0)

    def get_maxQ(self, observation):
        """
        Called when the agent is asked to find the maximum Q-value of
        all actions based on the 'observation' the environment is in.
        """
        self.create_Q_if_new_observation(observation)
        return max(self.Q[observation].values())

    def choose_action(self, observation, agent99, env):
        """
        Choose which action to take, based on the observation.
        If observation is seen for the first time, initialize its Q values to 0.0
        """
        if observation in agent99.Q:
          self.create_Q_if_new_observation(observation)
          maxQ = self.get_maxQ(observation)
          maxQ99 = agent99.get_maxQ(observation)

          Isdone11, Isdone99 = False, False
          while not Isdone11:
              action99 = random.choice([k for k in agent99.Q[observation].keys()
                                      if agent99.Q[observation][k] == maxQ99])
              _1, payout99, is_done, _3 = env.step(action99)
              Isdone11 = is_done

          while not Isdone99:
              action11 = random.choice([k for k in self.Q[observation].keys()
                                      if self.Q[observation][k] == maxQ])
              __1, payout, is_done, __3 = env.step(action11)
              Isdone99 = is_done

          if payout99 > payout:
            action = action99
          else:
            action = action11
        else:
            self.create_Q_if_new_observation(observation)

            # uniformly distributed random number > epsilon happens with probability 1-epsilon
            if random.random() > self.epsilon:
                maxQ = self.get_maxQ(observation)

                # multiple actions could have maxQ- pick one at random in that case
                # this is also the case when the Q value for this observation were just set to 0.0
                action = random.choice([k for k in self.Q[observation].keys()
                                        if self.Q[observation][k] == maxQ])
            else:
                action = random.choice(self.valid_actions)

        self.update_parameters()

        return action

#     def choose_action(self, observation, agent99):
#       if observation in agent99.Q:
#         self.create_Q_if_new_observation(observation)
#             maxQ = self.get_maxQ(observation)
#             maxQ99 = agent99.get_maxQ(observation)
#             if ( maxQ99 > maxQ):
#                   action = random.choice([k for k in agent99.Q[observation].keys()
#                                         if agent99.Q[observation][k] == maxQ99] )
#               else:
#                     action = random.choice([k for k in self.Q[observation].keys()
#                                           if self.Q[observation][k] == maxQ] )
#             else:
#                   self.create_Q_if_new_observation(observation)
#             if random.random() > self.epsilon:
#                   maxQ = self.get_maxQ(observation)
#                 action = random.choice([k for k in self.Q[observation].keys()
#                                       if self.Q[observation][k] == maxQ])
#             else:
#                   action = random.choice(self.valid_actions)
#
#
#         # uniformly distributed random number > epsilon happens with probability 1-epsilon
#         
#         self.update_parameters()
#
#         return action


    def learn(self, observation, action, reward, next_observation, other_agent):
        """
        Called after the agent completes an action and receives an award.
        This function does not consider future rewards
        when conducting learning.
        """

        # Q = Q*(1-alpha) + alpha(reward + discount * utility of next observation)
        # Q = Q - Q * alpha + alpha(reward + discount * self.get_maxQ(next_observation))
        # Q = Q - alpha (-Q + reward + discount * self.get_maxQ(next_observation))
        self.Q[observation][action] += self.alpha * (reward
                                                     + (self.gamma * other_agent.get_maxQ(next_observation))
                                                     - self.Q[observation][action])

import matplotlib
matplotlib.use('Agg')

import gym
import matplotlib.pyplot as plt

env = gym.make('Blackjack-v0')
env1 = gym.make('Blackjack-v0')
# env1=env

agent = Agent(env=env, epsilon=1.0, alpha=0.5, gamma=0.2, num_episodes_to_train=800)
agent1 = Agent(env=env1, epsilon=1.5, alpha=0.75, gamma=0.3, num_episodes_to_train=800)

num_rounds = 1000 # Payout calculated over num_rounds
num_samples = 1000 # num_rounds simulated over num_samples

average_payouts = []
average_payouts1 = []

observation = env.reset()
observation1 = env1.reset()
for sample in range(num_samples):
    round = 1
    total_payout = 0 # to store total payout over 'num_rounds'
    total_payout1 = 0 # to store total payout over 'num_rounds'
    # Take action based on Q-table of the agent and learn based on that until 'num_episodes_to_train' = 0
    while round <= num_rounds:
        action = agent.choose_action(observation, agent1, env1)
        next_observation, payout, is_done, _ = env.step(action)
        agent.learn(observation, action, payout, next_observation, agent1)
        total_payout += payout
        observation = next_observation

        action1 = agent1.choose_action(observation1, agent, env)
        next_observation1, payout1, is_done1, __ = env1.step(action1)
        agent1.learn(observation1, action1, payout1, next_observation1, agent)
        total_payout1 += payout1
        observation1 = next_observation1

        if is_done or is_done1:
            observation = env.reset() # Environment deals new cards to player and dealer
            observation1 = env1.reset() # Environment deals new cards to player and dealer
            round += 1
    average_payouts.append(total_payout)
    average_payouts1.append(total_payout1)

# Plot payout per 1000 episodes for each value of 'sample'
plt.plot(average_payouts)           
plt.xlabel('num_samples')
plt.ylabel('payout after 1000 rounds')
plt.plot(average_payouts1)
plt.savefig('test.png', dpi=(200))
    
print ("Average payout after {} rounds is {}".format(num_rounds, sum(average_payouts)/(num_samples)))
print ("Average payout 1 after {} rounds is {}".format(num_rounds, sum(average_payouts1)/(num_samples)))

env.close()
env1.close()
