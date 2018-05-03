# -*- coding: utf-8 -*-

# AI for Self Driving Car

# Importing the libraries

import numpy as np
import random
import os
import sys
import torch # arrays on GPU
import torch.nn as nn # Neural Network linrary
import torch.nn.functional as F
import torch.optim as optim # optimizer for stochastic gradient descent
import torch.autograd as autograd
from torch.autograd import Variable # import Variable class

# Architecture of the Neural Network

class Network(nn.Module):  # subclass of nn.Module

    def __init__(self, input_size, output_size):  # input neurons 5, output neurons 3
        super(Network, self).__init__()  # pytorch trick #always call parent's init
        self.input_size = input_size
        self.output_size = output_size
        self.fc1 = nn.Linear(input_size, 24)  # full connection 1 (input layer -> hidden)
        #self.fc2 = nn.Linear(24, 24)          # full connection 2 (2nd hidden layer)
        self.fc3 = nn.Linear(24, output_size)  # full connection 3 (hidden -> output)

    def forward(self, state):  # activate neurons, perform propagation
        '''
            Pass an input through the layer,
            Perform operations on inputs using parameters,
            and return the output.
        '''
        x = F.relu(self.fc1(state)) # x : hidden neurons, relu : rectifier function, we want actived hidden neurons x
        #y =  F.relu(self.fc2(x))
        q_values = self.fc3(x) # output neurons are the q values for each possible actions
        return q_values


# Implementing Experience Replay

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        
    def push(self, event):
        self.memory.append(event)
        if len(self.memory) > self.capacity:
            del self.memory[0]

    def sample(self, batch_size):
        samples = zip(*random.sample(self.memory, batch_size))
        return map(lambda x: Variable(torch.cat(x, 0)), samples)

# Implementing Deep Q Learning

BATCH_SIZE = 32 # 100
MEMORY_CAPACITY = 2000 # 100000
printTimeToLearn = False

class Dqn(object):

    def __init__(self, input_size, nb_action, gamma=0.9):  # gamma is the delay coefficient
        self.gamma = gamma
        self.reward_window = []  # mean of the reward over time
        self.model = Network(input_size, nb_action)  # create network
        # model(state) is callable, call forward internally -> get ouputs
        self.memory = ReplayMemory(MEMORY_CAPACITY)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001) # optimizer for stochastic gradient descent # connect it to our NN
        self.last_state = torch.Tensor(input_size).unsqueeze(0) # need to be a tensor, with in addition a fake dimension at the beginning corresponding to the batch
        self.last_action = 0 # init to action 0
        self.last_reward = 0
        self.brain_file = 'last_brain.pth' # name of the file we save NN and optimizer parameters

    def select_action(self, state):
        '''
            Select action base on outputs Q values of NN, that depends from the input state
            Action selection policy : Softmax to explore more solutions
        '''
        T = 80 # temperature parameter -> influence probability => affect selection of action
        # closer to 0, the less sure it is when playing an action, the higher the more sure
        # e.g. T=3 softmax([1,2,3]) = [0.04,0.11,0.85] => softmax([1,2,3] * 3 ) = [0,0.2,0.98]
        probs = F.softmax(self.model(Variable(state, volatile =True) * T))
        # state is a torch tensor. most tensor are wrapped into torch Variable that also contain gradient
        # we dont want the gradient in the graph of computation => then volatile = True -> save a lot of memory, improve performance
        action = probs.multinomial() # random draw from the probibality distribution
        return action.data[0,0] # action is a Tensor of size 1x1 -> get element at index at [0,0]

    def learn(self, batch_state, batch_next_state, batch_action, batch_reward):
        '''
            Train the deep neural network. Whole process of forward/backward propagation.
            Get output, get target, compare target to output to compute last error and backpropagate.
            Update weights using stochastic gradient descent.
        '''
        # our model is excecting a batch of input states so we can use batch_state
        # we dont want output of all possible actions, we are interested in the actions that were chosen
        # the actions choosen by nn to play at each time -> so we use gather function
        # we gather each time the best action to play for each of the input states of batch state
        # add fake dimension to batch_action (at 1 ) to make it compatible to batch_state
        # then kill fake dimension to make it tensor
        outputs = self.model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1) # prediction of Q(s,a)
        next_outputs = self.model(batch_next_state).detach().max(1)[0] # next_Q(s,a)
        target = self.gamma * next_outputs + batch_reward # target = gamma * next_Q(s,a) + R
        td_loss = F.smooth_l1_loss(outputs, target) # get the loss (error of NN prediction) comparing outputs to target
        self.optimizer.zero_grad() # reinitialize optimizer at each iteration of stoch grad descent
        td_loss.backward(retain_variables=True) # backward propagation of error in network # retain_variables = true -> free some memory and improve perf
        self.optimizer.step() # update the weights according to backward prop using optimizer

    def update(self, reward, signal):
        global printTimeToLearn

        new_state = torch.Tensor(signal).float().unsqueeze(0) # convert to Tensor and add fake dim
        event = (self.last_state, new_state, torch.LongTensor([int(self.last_action)]), torch.Tensor([self.last_reward])) # current experience
        
        self.memory.push(event) # add to the memory
        action = self.select_action(new_state) # get the action
        
        if len(self.memory.memory) > 100: # more than 100 events in memory -> time to learn
            if not isPrinted:
                print('time to learn')
                isPrinted = True
            batch = self.memory.sample(BATCH_SIZE)
            batch_state, batch_next_state, batch_action, batch_reward = batch # get data samples (index 1)
            self.learn(batch_state, batch_next_state, batch_action, batch_reward)
                
        self.last_action = action
        self.last_state = new_state
        self.last_reward = reward
        self.reward_window.append(reward)
        if len(self.reward_window) > 1000:
            del self.reward_window[0]
        return action

    def score(self):
        '''
            Score is the mean of the reward in the sliding window
        '''
        score = sum(self.reward_window)/(len(self.reward_window)+1.) # +1 to avoid division by zero
        return score

    def save(self):
        '''
            Save Neural Network and optmizer. (last weights)
        '''
        torch.save({'state_dict' : self.model.state_dict(),
                    'optimizer' : self.optimizer.state_dict(),
                    }, self.brain_file)

    def load(self):
        '''
            Load brain_file that contains saved Neural Network and optmizer weights
        '''
        if os.path.isfile(self.brain_file):
            print("=> loading checkpoint...")
            checkpoint = torch.load(self.brain_file)
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("done !")
        else:
            print("no checkpoint found...")
