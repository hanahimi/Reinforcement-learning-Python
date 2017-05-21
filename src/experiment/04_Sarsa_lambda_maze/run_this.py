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
from RL_brain import SarsaLambdaTable

def update():
    for episode in range(100):
        # 更新窗口环境
        observation = env.reset()
#         RL.reset_trace()
        action = RL.choose_action(str(observation))

        while True:
            # fresh env
            env.render()

            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)
            
            # RL choose action based on next observation
            action_ = RL.choose_action(str(observation_))
            
            # RL learn from this transition (s, a, r, s, a) ==> Sarsa
            RL.learn(str(observation),action, reward, str(observation_),action_)
            
            # swap observation and action
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
    RL = SarsaLambdaTable(actions=list(range(env.n_actions)))
    
    env.after(100, update)
    env.mainloop()
    
    
    