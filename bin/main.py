import policy
import actor
import dqn
from config import *


"""  This is for training 

## 
more detailed parameter adaptations can be made in config.py
However! be aware that config.py may be changed during the execution of main.py
##

-- Games --
    'Pong-v0'
    
-- Architectures --
    'CNN'
    'FC'

-- Algorithms --
    'policy'    # policy learning
    'dqn'       # deep q-learning
    'actor'     # actor critic learning



"""

# training Schedule
i = 0
args = Args()
#set static args
args.max_episodes = 2000

while True:
    args.saving_prefix = "experiment_" + str(i + 1)
    # Setup - training schedule
    if i == 0:
        # change specific parameters here
        # args.setGame('Pong-v0')
        args.setGame('BeamRider-v0')
        args.p_arch = 'CNN'
        args.p_learning_rate = 1e-3
        policy.main(args)


    #elif i = ... further scheduling
    else:
        print("Schedule completed")
        exit()

    i = i + 1