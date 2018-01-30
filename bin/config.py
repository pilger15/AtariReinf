"""

###### DO NOT CHANGE PARAMETERS HERE use main.py#######
-- Games --
    'Pong-v0'

-- Architectures --
    'CNN'
    'FC'

"""
class Args():
    # Game-specific
    game = None
    action_dict = None
    n_action = None

    def setGame(self, setGame):
        self.game = setGame
        if setGame == "Pong-v0":
            self.action_dict = {0: 2, 1: 3}
            self.n_action = 2

        if setGame == "BeamRider-v0":
            self.action_dict = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8}
            self.n_action = 8

    env = None

    # Architecture specifics
    h, w = [105, 80]

    p_arch = 'CNN'
    q_arch = 'CNN'

    q_learning_rate = 1e-4
    p_learning_rate = 1e-3

    state_length = 1    # if state length = 1 the difference between the last two frames is the state

    # training specifics
    home_dir = '..'
    max_episodes = 0  # 0-> no limits
    saving_prefix = "EXP1"   # prevent overwriting

    gamma = 0.99
    start_episode = 0      # TODO implement resume
    mini_batch_size = 32
    batch_size = 10  # every how many points to do a param update?
    save_interval = 50

    #dqn specific
    replay_capacity = pow(2, 15)


    def printArgs(self):
        pass
