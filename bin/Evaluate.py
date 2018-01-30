import gym
from utils import *
import os
from models import *
from config import *
import numpy as np

""" This is for evaluation  """


args = Args()
# args.setGame('Pong-v0')
args.setGame('BeamRider-v0')


def main():

    n_games = 15  # number of games played for each model-checkpoint

    model_type = 'policy'
    saving_prefix = "experiment_0"  # prevent overwriting

    evaluate_dir = args.home_dir+('/models/%s/%s/%s/' % (args.game, saving_prefix, model_type))
    print(evaluate_dir)

    model = FC(args).cuda()
    state = State(args.state_length)

    env = None

    logfile = open(evaluate_dir +'eval.csv', 'w+')
    logfile.close()

    for file in sorted(os.listdir(evaluate_dir)):
        filepath = ('%s/%s' %(evaluate_dir, file))
        if file.endswith('.pkl'):
            logfile = open(evaluate_dir + 'eval.csv', 'a+')

            checkpoint = load_checkpoint_from_file(filepath)
            model.load_state_dict(checkpoint['model_dict'])
            model.eval()

            if not env:
                print('==> Initializing game: %s \n' % (checkpoint['game']))
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
                elif model_type == 'dqn':
                    sampled_action = torch.max(model.action_values(Variable(s, volatile=True)), dim=-1)[1].data
                else:
                    print('Model identifier unrecognized')
                    exit()
                observation, reward, done, info = env.step(args.action_dict[sampled_action[0]])  # Get Reward for action

                rewards.append(reward)

                if done:
                    games_played += 1
                    observation = env.reset()
                    state.clear()
                    scores.append(np.sum(rewards))
                    rewards = []

            s_means = np.mean(scores)
            s_std = np.std(scores)
            print('Episode [%s] evaluated | Mean: [%f+-%f] | Training time: [%fs]\n' %(checkpoint['episode'], s_means, s_std, checkpoint['time']))
            logfile.write('%d,%f,%f,%f\n'%(checkpoint['episode'], s_means, s_std, checkpoint['time']))
            logfile.close()

    print('\nEvaluation done.')
if __name__ == '__main__':
    main()
