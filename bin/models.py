import torch.nn as nn
import torch.nn.functional as F
from config import *


class FC(nn.Module):
    """
    Fully connected network
    Important : Either trained for action probability prediction (policy gradient) or q value prediction (dqn algorithm),
    this means there is no semantic relation softmax(q_value) = action probabilites!
    """
    arch = 'FC'
    def __init__(self, args=Args()):
        super(FC, self).__init__()
        self.args = args
        self.h1 = nn.Linear(self.args.state_length * self.args.h * self.args.w, 200)
        self.h2 = nn.Linear(200, self.args.n_action)


    def policy(self, x):
        """
        foreward for policy algorithm.

        :param x: -> observation frame statelength * h * w
        :return: action-probability
        """
        x = self.action_values(x)
        return F.softmax(x, dim=-1)

    def action_values(self, x):
        """
        foreward for dqn algorithm.

        :param x: -> observation frame statelength * h * w
        :return: q-values
        """
        x = F.relu(self.h1(x.view(-1, self.args.state_length * self.args.h * self.args.w).float()))
        x = self.h2(x)
        return x


class CNN(nn.Module):
    """
     CNN
     Important : Either trained for action probability prediction (policy gradient) or q value prediction (dqn algorithm),
     this means there is no semantic relation softmax(q_value) = action probabilites!
     """

    arch = 'CNN'

    def __init__(self, args=Args()):
        super(CNN, self).__init__()
        self.args = args

        self.c1 = nn.Conv2d(self.args.state_length, 16, (8, 8), stride=4)
        self.c2 = nn.Conv2d(16, 16, (4, 4), stride=2)
        self.l1 = nn.Linear(11 * 8 * 16, 128)
        self.l2 = nn.Linear(128, self.args.n_action)


    def policy(self, x):
        """
        foreward for policy algorithm.

        :param x: -> observation frame statelength * h * w
        :return: action-probability
        """
        x = self.action_values(x)
        return F.softmax(x, dim=-1)

    def action_values(self, x):
        """
        foreward for dqn algorithm.

        :param x: -> observation frame statelength * h * w
        :return: q-values
        """
        x = F.relu(self.c1(x.view(-1, self.args.state_length, self.args.h, self.args.w).float()))
        x = F.relu(self.c2(x))
        x = F.relu(self.l1(x.view(-1, 11 * 8 * 16)))
        x = self.l2(x)
        return x
