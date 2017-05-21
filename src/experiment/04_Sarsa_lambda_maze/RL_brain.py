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

# on-policy
class SarsaLambdaTable(RL):
    def __init__(self, actions,
                 learning_rate=0.01,
                 reward_decay=0.9,
                 e_greedy=0.9,
                 trace_decay=0.9):
        super(SarsaLambdaTable, self).__init__(actions, learning_rate, reward_decay, e_greedy)
        # 后向观测算法
        # eligibility trace 用于为每个状态和行为进行计数
        self.lambda_ = trace_decay
        self.eligibility_trace = self.q_table.copy()
    
    def check_state_exist(self, state):
        """ 新增检查eligibility trace的新状态"""
        if state not in self.q_table.index:
            # 使用全0的行进行新状态的填充
            to_be_append = pd.Series(
                    [0]*len(self.actions),
                    index = self.q_table.columns,
                    name = state)
            self.q_table = self.q_table.append(to_be_append)
            self.eligibility_trace = self.eligibility_trace.append(to_be_append)
    
    def learn(self, s, a, r, s_, a_):
        # 这部分和Sarsa一样
        self.check_state_exist(s_)
        q_predict = self.q_table.ix[s, a]
        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.ix[s_, a_]
        else:
            q_target = r
        err = self.lr * (q_target - q_predict)
        
        # 不同
        # 对于经历过的state-action，让他+1,证明他是在得到reward的途中不可或缺的一环
        # Method 1:
        self.eligibility_trace.ix[s, a] += 1
        
        # Method 2: (更有效的方法)
        # 强调下次获得的奖励只和在这个 state, 这次选的 a 有关
        self.eligibility_trace.ix[s, :] *= 0
        self.eligibility_trace.ix[s, a] = 1

        """
        For all s in S, a in A(s):
            Q(s,a) <- Q(s,a) + lr * err * E(s,a)
            E(s,a) <- gamma * lambda * E(s,a)
        """
        # Q table 更新(由于q_table和行列和trace一致，因此可以按元素对应运算)
        self.q_table += self.lr * err * self.eligibility_trace
        # 随着时间衰减 eligibility trace 的值, 
        # 离获取 reward 越远的步, 他的"不可或缺性"越小
        self.eligibility_trace *= self.gamma * self.lambda_
        
    def reset_trace(self):
        """
        eligibility trace 只是记录每个回合的每一步, 
        新回合开始的时候需要将 Trace 清零.
        """
        self.eligibility_trace *= 0
    
if __name__=="__main__":
    pass
    
    