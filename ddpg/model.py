
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

from functools import reduce # Valid in Python 2.6+, required in Python 3
import operator

def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)

class BaseModel(nn.Module):
    def __init__(self, nb_states, nb_actions, init_w):
        super(BaseModel, self).__init__()
        self.nb_states = nb_states
        self.nb_actions = nb_actions
        self.init_w = init_w

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

        in_dims = self.nb_states[0]
        self.conv1 = nn.Conv2d(in_dims, 32, 5, 1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 32, 5, 1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 64, 4, 1)
        self.cnn_out_dims = self._get_cnn_out_dims()

    def init_weights(self, init_w):
        self.conv1.weight.data = fanin_init(self.conv1.weight.data.size())
        self.conv2.weight.data = fanin_init(self.conv2.weight.data.size())
        self.conv3.weight.data = fanin_init(self.conv3.weight.data.size())

    def forward(self, x):
        batch_size = x.data.size()[0]
        out = self._forward_cnn(x)
        out = out.view(batch_size, -1)
        return out
    
    def _forward_cnn(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.pool1(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.pool2(out)
        out = self.conv3(out)
        out = self.relu(out)
        return out

    def _get_cnn_out_dims(self):
        x = Variable(torch.rand(self.nb_states).view((1,) + self.nb_states))
        out = self._forward_cnn(x).data.numpy()
        return int(np.prod(out.shape[1:]))

    def _sample_output(self, batch_size):
        # For debugging
        x = Variable(torch.rand((batch_size,) + self.nb_states))
        out = self.forward(x).data.numpy()
        return out 

class Actor(BaseModel):
    def __init__(self, nb_states, nb_actions, init_w=3e-4):
        super(Actor, self).__init__(nb_states, nb_actions, init_w)

        self.fc1 = nn.Linear(self.cnn_out_dims, 200)
        self.fc2 = nn.Linear(200, nb_actions)

        self.init_weights(init_w)
    
    def init_weights(self, init_w):
        super(Actor, self).init_weights(init_w)
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data.uniform_(-init_w, init_w)
        self.fc2.bias.data.uniform_(-init_w, init_w)
    
    def forward(self, x):
        cnn_out = super(Actor, self).forward(x)
        out = self.fc1(cnn_out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.tanh(out)
        return out
    
class Critic(BaseModel):
    def __init__(self, nb_states, nb_actions, init_w=3e-4):
        super(Critic, self).__init__(nb_states, nb_actions, init_w)

        self.fc1 = nn.Linear(self.cnn_out_dims + nb_actions, 200)
        self.fc2 = nn.Linear(200, 1)

        self.init_weights(init_w)
        
    def init_weights(self, init_w):
        super(Critic, self).init_weights(init_w)
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data.uniform_(-init_w, init_w)
        self.fc2.bias.data.uniform_(-init_w, init_w)
    
    def forward(self, xs):
        x, a = xs
        cnn_out = super(Critic, self).forward(x)
        merge_out = torch.cat([cnn_out, a], 1)
        out = self.fc1(merge_out)
        out = self.relu(out)
        out = self.fc2(out)
        return out

    def _sample_output(self, batch_size):
        # For debugging
        x = Variable(torch.rand((batch_size,) + self.nb_states))
        a = Variable(torch.rand((batch_size, self.nb_actions)))
        out = self.forward((x, a)).data.numpy()
        return out 
