import numpy as np
import collections
import torch
from torch.autograd import Variable
import os
from timeit import default_timer as timer
import numpy as np
from config import *

args = Args()
class PolicyMemory:
    """
    memory for the policy gradient algorithm
    memory is only used for one gradient update ( different from replay memory!)
    save_q_baseline is needed only in actor critic mode and
    """
    def __init__(self, gamma, save_q_baseline=False, normalize_reward=False):
        """
        :param gamma: discount factor
        :param save_q_baseline: saves the state value estimated by the dqn -> only needed for actor critic
        :param normalize_reward: normalizes the discounted reward (do not use together with Baseline class defined in utils!)
        """
        self.gamma = gamma
        self.save_baseline = save_q_baseline
        self.normalize_reward = normalize_reward
        self.memory_length = 0
        self.n_rewards = 0                               #number of non zero rewards in memory
        self.observation_memory = collections.deque()
        self.action_memory = collections.deque()
        self.reward_memory = collections.deque()
        self.terminal_memory = collections.deque()
        self.est_state_val_memory = collections.deque()

    def remember(self, observation, action, reward, terminal, est_state_val):
        """
        save games states for gradient updates
        """
        self.observation_memory.append(observation)
        self.action_memory.append(action)
        self.reward_memory.append(reward)
        self.terminal_memory.append(terminal)
        self.memory_length += 1
        if terminal:
            self.n_rewards += 1
        if self.save_baseline:
            self.est_state_val_memory.append(est_state_val)

    def discounted_reward(self):
        """
        calculate the discounted rewards. Discount will not be propagated through 'terminal' boundary
        running reward will be normalized if 'normalize_reward' is set
        :return: discounted rewards
        """
        discounted_reward = np.zeros_like(self.reward_memory)
        temp_sum = 0.
        for t in reversed(range(0, self.memory_length)):
            if self.terminal_memory[t]:                                                                                                     #block discounted reward propagation from other game instances
                temp_sum = 0.
            temp_sum = self.reward_memory[t] + temp_sum * self.gamma
            discounted_reward[t] = temp_sum
        if self.normalize_reward:
            discounted_reward -= np.mean(discounted_reward)
            discounted_reward /= np.std(discounted_reward + 1e-6)
        return discounted_reward

    def sample_batch(self):
        """
        estimated state value is empty in policy gradient mode !
        returns the saved observations, actions, discounted reward and estimated state value
        which are needed for the policy gradient update (look at policy.py)
        after returning the memory the memory is deleted
        """
        disc_rew = self.discounted_reward()
        observation_batch = Variable(torch.stack(self.observation_memory)).cuda()
        discounted_reward_batch = Variable(torch.FloatTensor(disc_rew).view(-1, 1)).cuda()
        action_batch = Variable(torch.stack(self.action_memory)).cuda()
        est_state_val_batch = Variable(torch.stack(self.est_state_val_memory)).cuda() if self.save_baseline else []
        self.forget()
        return observation_batch, action_batch, discounted_reward_batch, est_state_val_batch

    def forget(self):
        """
        reset the memory
        """
        self.observation_memory = collections.deque()
        self.action_memory.clear()
        self.reward_memory.clear()
        self.terminal_memory.clear()
        self.est_state_val_memory.clear()
        self.memory_length = 0
        self.n_rewards = 0

class DQNMemory:
    """
       replay memory for the dqn
       replay memory is filled up until it reaches replay_capacity then the oldest entries are replaced with new entries
       """
    def __init__(self, replay_capacity=pow(2, 15)):
        """
        :param replay_capacity: maximum replay memory length
        """
        self.replay_capacity = replay_capacity
        self.memory_length = 0
        self.history_length = 1
        self.obsv_memory = collections.deque(maxlen=replay_capacity)  # Replay Memory
        self.reward_memory = collections.deque(maxlen=replay_capacity)
        self.action_memory = collections.deque(maxlen=replay_capacity)
        self.terminal_memory = collections.deque(maxlen=replay_capacity)

    def remember(self, observation, action, reward, is_terminal):
        """
        save games states for multiple gradient updates
        """
        self.obsv_memory.append(observation)
        self.action_memory.append(action)
        self.reward_memory.append(reward)
        self.terminal_memory.append(is_terminal)
        if self.memory_length < self.replay_capacity:
            self.memory_length += 1

    def sample_mini_batch(self, mini_batch_size):
        """return a mini batch from replay memory"""
        sample_idx = np.random.randint(self.history_length,  self.memory_length - 1, size=mini_batch_size)  # sample idx
        action_batch, reward_batch, t_1_batch, t_2_batch, terminal_batch = [], [], [], [], []
        for idx in sample_idx:
            reward_batch.append(self.reward_memory[idx])
            action_batch.append(self.action_memory[idx])
            t_1_batch.append(self.obsv_memory[idx])
            t_2_batch.append(self.obsv_memory[idx + 1])
            terminal_batch.append(self.terminal_memory[idx])

        reward_batch = torch.FloatTensor(reward_batch).cuda()
        terminal_batch = torch.ByteTensor(terminal_batch).cuda()
        action_batch = Variable(torch.stack(action_batch, dim=0)).cuda()
        t_1_batch = Variable(torch.stack(t_1_batch, dim=0)).cuda()
        t_2_batch = Variable(torch.stack(t_2_batch, dim=0), volatile=True).cuda()
        return t_1_batch, t_2_batch, reward_batch, action_batch, terminal_batch

class Baseline():
    """
    returns the avarage discounted reward of the last @p window_len @p n_rewards
    """
    def __init__(self, window_len=100):
        self.window_len = window_len
        self.rewards = collections.deque(maxlen=window_len)

    def get(self):
        return torch.mean(torch.FloatTensor(self.rewards).cuda())

    def append(self, reward):
        self.rewards.append(torch.mean(reward))



class State:
    """
    returns the state
    if @p state_length is 1 the frame difference of the last frames is returned in one channel
     if @p state_length > 1 the last @p state_length are concatenated in the channel dimension
    """
    def __init__(self, length):
        self.length = length
        self.frame_memory = collections.deque(maxlen=length)
        self.clear()

    def clear(self):
        for i in range(self.length):
            self.frame_memory.append(torch.zeros(args.h, args.w).float().cuda())

    def get_state(self, frame):

        if self.length == 1:
            return self.frame_difference(frame)
        else:
            frame = torch.FloatTensor(self.preprocessing(frame)).cuda()
            self.frame_memory.append(frame)
            state = torch.stack(self.frame_memory, dim=0)
            return state

    def frame_difference(self, frame):
        frame = torch.FloatTensor(self.preprocessing(frame)).cuda()
        state = frame - self.frame_memory[0]
        self.frame_memory.append(frame)
        return state

    def preprocessing(self, img):
        return np.mean(img[::2, ::2], axis=2) / 127.5 - 1.


def startTimer():
    return timer()

def resetTimer(start):
    return timer() - start, timer()

def epsilon_decay(episode):
    if episode < 1000:
        return 0.8 * (1. - episode/1000.) + 0.1
    else:
        return 0.1

def load_checkpoint_from_file(path):
    """
    loads a checkpoint from path. Use "load_checkpoint" function to load from parameters
    :param path: string -> filepath of the model to be loaded
    :return: checkpoint.pkl
    """
    return torch.load(path)

def load_checkpoint(saving_prefix, game, model_type, from_episode):
    #TODO implement load_checkpoint version below might be deprecated
    """
    loads a checkpoint by generating the path from input parameters
    :param saving_prefix: string -> a prefix to separate different training runs
    :param game: string -> identifier string of the game environment
    :param model_type: string -> identify the algorithm
    :param from_episode: int -> the episode at which the model was saved
    :return:
    """
    path = '%s/models/%s/%s/%s' % (args.home_dir, game, saving_prefix, model_type)
    path = '%s/e%d.pkl' % (path, from_episode)
    print('Loading checkpoint from %s' % path)
    return load_checkpoint_from_file(path)



def save_checkpoint(saving_prefix, game, algorithm, model, lr, episode, time, save_args):
    filename = '%s/models/%s/%s/%s' % (save_args.home_dir, game, saving_prefix, algorithm)
    if not os.path.exists(filename):
        os.makedirs(filename)

    filename = '%s/e%d.pkl' % (filename, episode)
    print('\tSaving checkpoint to [%s] ...' % filename)

    state = {
        'game': game,
        'episode': episode,
        'time': time,
        'algorithm': algorithm,
        'model_dict': model.state_dict(),
        'arch': model.arch,
        'lr': lr,
        'preprocessing': 'difference' if args.state_length == 1 else 'frame_history',
        'args': save_args
    }

    torch.save(state, filename)

    print('\t...Saving completed')
