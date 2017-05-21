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
from RL_brain import SarsaTable

def update():
    for episode in range(100):
        # 更新窗口环境
        observation = env.reset()
        # Sarsa 会先做出决策
        action = RL.choose_action(str(observation))
        while True:
            # 刷新窗口状态
            env.render()
            
            # 根据当前选择a，观测当前的奖励和下一个状态的情况
            observation_, reward, done = env.step(action)
            
            # 根据Q表的策略对下一个观测进行选择
            action_ = RL.choose_action(str(observation_))
            
            
            # RL算法进行学习，使用Sarsa算法(s,a,r,s_,a_)更新当前的Q(s,a)
            RL.learn(str(observation),action, reward, str(observation_),action_)
            
            # 状态转移
            observation = observation_
            action = action_
            
            # 当找到宝藏或掉坑里终止学习过程
            if done:
                break

    print("game over")
    env.destroy()

            
if __name__=="__main__":
    pass
    env = Maze()
    RL = SarsaTable(actions=list(range(env.n_actions)))
    
    env.after(100, update)
    env.mainloop()
    
    
    