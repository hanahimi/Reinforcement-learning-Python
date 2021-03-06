#-*-coding:UTF-8-*-
'''
Created on 2017年5月16日
@author: Ayumi Phoenix

Reinforcement learning maze example.

Red rectangle:          explorer.
Black rectangles:       hells       [reward = -1].
Yellow bin circle:      paradise    [reward = +1].
All other states:       ground      [reward = 0].

This script is the main part which controls the update method of this example.
The RL is in RL_brain.py.

ref: https://morvanzhou.github.io
'''
from maze_env import Maze
from RL_brain import QLearningTable

def update():
    for episode in range(100):
        # 更新窗口环境
        observation = env.reset()
        while True:
            # 刷新窗口状态
            env.render()
            
            # 根据Q表的策略基于当前观测进行选择
            action = RL.choose_action(str(observation))
            
            # 根据当前选择a，观测当前的奖励和下一个状态的情况
            observation_, reward, done = env.step(action)
            
            # RL算法进行学习，使用Qlearning算法(s,a,r,s_)
            RL.learn(str(observation),action, reward, str(observation_))
            
            # 状态转移
            observation = observation_
            
            # 当找到宝藏或掉坑里终止学习过程
            if done:
                break

    print("game over")
    env.destroy()

            
if __name__=="__main__":
    pass
    env = Maze()
    RL = QLearningTable(actions=list(range(env.n_actions)))
    
    env.after(100, update)
    env.mainloop()
    
    
    