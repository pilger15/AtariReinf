import gym
from torch.distributions import Categorical
from utils import *
import os
from models import *
from config import *
import numpy as np
args = Args()

def main():


    model_type = 'policy'
    saving_prefix = "experiment_5"  # prevent overwriting
    from_episode_number = 100

    evaluate_dir = args.home_dir+('/models/Pong-v0/%s/%s/' %(saving_prefix, model_type))
    # evaluate_dir = '/home/dominik/Desktop/stud/AI-Projekt/models/Pong-v0/policy_with_unkown_parameters/'

    model = FC().cuda()
    state = State(args.state_length)

    env = None

    filepath = ('%s/e%d.pkl' %(evaluate_dir,from_episode_number))

    checkpoint = load_checkpoint_from_file(filepath)
    model.load_state_dict(checkpoint['model_dict'])
    model.eval()
    # action_dict = getActionDict(game)
    if not env:
        env = gym.make(checkpoint['game'])

    print('Evaluating %s' %file)

    observation = env.reset()
    games_played = 0

    model.eval()
    scores = []
    rewards = []

    while games_played in range(n_games):
        # env.render()
        s = state.get_state(observation)
        if model_type == 'policy':
            sampled_action = torch.max(model.policy(Variable(s, volatile=True)), dim=-1)[1].data
        elif model_type == 'dqn':           #todo this is only temporary for DQN becuase it was saved incorrectly as cnn type
            sampled_action = torch.max(model.action_values(Variable(s, volatile=True)), dim=-1)[1].data

        observation, reward, done, info = env.step(args.action_dict[sampled_action[0]])  # Get Reward for action


        if done:
            games_played += 1
            observation = env.reset()
            state.clear()


        print('Episode [%s] evaluated | Mean: [%f+-%f] | Training time: [%fs]\n' %(checkpoint['episode'], s_means, s_std, checkpoint['time']))

    print('\nEvaluation done.')
if __name__ == '__main__':
    main()
