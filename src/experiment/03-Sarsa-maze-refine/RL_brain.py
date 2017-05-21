#-*-coding:UTF-8-*-
'''
Created on 2017年5月17日
@author: Ayumi Phoenix
总结Q-learning 和 Sarsa的相同行为，用RL作为父类
建立继承关系
'''

import numpy as np
import pandas as pd

class RL(object):
    def __init__(self, action_space, learning_rate=0.01,reward_decay=0.9,e_greedy=0.9):
        self.actions = action_space  # a list
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy

        self.q_table = pd.DataFrame(columns=self.actions)

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(
                pd.Series(
                    [0]*len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )
    
    def choose_action(self, observation):
        self.check_state_exist(observation)
        # action selection
        if np.random.rand() < self.epsilon:
            # choose best action
            state_action = self.q_table.ix[observation, :]
            state_action = state_action.reindex(np.random.permutation(state_action.index))     # some actions have same value
            action = state_action.argmax()
        else:
            # choose random action
            action = np.random.choice(self.actions)
        return action
    
    def learn(self, *args):
        pass

# off-policy
class QLearningTable(RL):
    def __init__(self, actions,learning_rate=0.01,reward_decay=0.9,e_greedy=0.9):
        super(QLearningTable, self).__init__(actions, learning_rate, reward_decay, e_greedy)
    
    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)
        q_predict = self.q_table.ix[s, a]
        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.ix[s_,:].max()
        else:
            q_target = r
        self.q_table.ix[s, a] += self.lr * (q_target - q_predict)

# on-policy
class SarsaTable(RL):
    def __init__(self, actions,learning_rate=0.01,reward_decay=0.9,e_greedy=0.9):
        super(SarsaTable, self).__init__(actions, learning_rate, reward_decay, e_greedy)
    
    def learn(self, s, a, r, s_, a_):
        self.check_state_exist(s_)
        q_predict = self.q_table.ix[s, a]
        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.ix[s_, a_]
        else:
            q_target = r
        self.q_table.ix[s, a] += self.lr * (q_target - q_predict)


if __name__=="__main__":
    pass
    
    