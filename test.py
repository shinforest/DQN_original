import os
import shutil
import random
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from collections import deque
import os
import shutil
import random
from collections import deque
from grid_world_one_hot import Grid
from DQN_Net import Policy

grid=Grid(4,3)
policy=Policy(12,4)
grid.draw_board()
t_action=grid.t_action

print("TEST=========================-")
test_state =  [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
policy.test_model(test_state)

grid.resets()
state=test_state
done=grid.done
actions=[]
print(state)
while not done:
    action=policy.choose_action(state)
    observation=grid.step(action)
    n_state=observation[0]
    reward=observation[1]
    done=observation[2]
    state=n_state
    actions.append(t_action[action])
    print(reward)
print("Result:",actions)
