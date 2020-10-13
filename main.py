"""
Dependencies:
tensorflow r1.2
keras 2.2.4
"""
import os
import shutil
import random
import numpy as np
import tensorflow as tf
from collections import deque
from grid_world_one_hot import Grid
from DQN_Net import Policy
grid=Grid(4,3)
policy=Policy(12,4)
grid.draw_board()
t_action=grid.t_action
r_log = []
for _ in range(10):
    grid.resets()
    state=grid.state
    done=grid.done
    actions=[]
    while not done:
        action=policy.choose_action(state)
        observation=grid.step(action)
        n_state=observation[0]
        reward=observation[1]
        done=observation[2]
        policy.learn_act(state,reward,n_state)
        state=n_state
        actions.append(t_action[action])
    r_log.append(reward)
    print(_,actions)

np.save("r_log", r_log)
policy.save_model()




# print("TEST=========================-")
# test_state =  [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
# policy.test_model(test_state)

# grid.resets()
# state=test_state
# done=grid.done
# actions=[]
# print(type(state))
# while not done:
#     action=policy.choose_action(state)
#     observation=grid.step(action)
#     n_state=observation[0]
#     reward=observation[1]
#     done=observation[2]
#     state=n_state
#     actions.append(t_action[action])
# print("Result:",actions)


