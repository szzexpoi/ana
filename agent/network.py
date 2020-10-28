import json
import math

import h5py
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

import torchvision.models as models


def compare_models(model_1, model_2):
    models_differ = 0
    for key_item_1, key_item_2 in zip(model_1.state_dict().items(), model_2.state_dict().items()):
        if torch.equal(key_item_1[1], key_item_2[1]):
            pass
        else:
            models_differ += 1
            if (key_item_1[0] == key_item_2[0]):
                print('Mismtach found at', key_item_1[0])
            else:
                raise Exception
    if models_differ == 0:
        print('Models match perfectly! :)')


class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)
        self.head = nn.Linear(448, 2)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))


class word2vec(nn.Module):
    """word2vec network (modified baseline with word embedding as target)
    """

    def __init__(self, method, mask_size=5):
        super(word2vec, self).__init__()
        self.word_embedding_size = 300
        # Observation layer
        self.fc_observation = nn.Linear(2048, 512)
        self.fc_target = nn.Linear(self.word_embedding_size, self.word_embedding_size)
        # Merge layer
        self.fc_merge = nn.Linear(
            self.word_embedding_size+512, 512)

    def save_gradient(self, grad):
        self.gradient = grad

    def forward(self, inp):
        # x is the observation
        # y is the target
        # z is the object location mask
        (x, y, z) = inp

        # observation
        x = x.view(512,49,-1).mean(1).view(-1)
        x = self.fc_observation(x)
        x = F.relu(x, True)
        
        # target
        y = y.view(-1)
        y = self.fc_target(y)
        y = F.relu(y, True)

        xy = torch.cat([x,y])
        xy = self.fc_merge(xy)
        xy = F.relu(xy, True)
        return xy

class word2vec_noconv(nn.Module):
    """Our method network without convolution for similarity grid
    """

    def __init__(self, method, mask_size=5):
        super(word2vec_noconv, self).__init__()
        self.word_embedding_size = 300
        self.fc_target = nn.Linear(
            self.word_embedding_size, self.word_embedding_size)
        # Observation layer
        self.fc_observation = nn.Linear(8192, 512)

        self.flat_input = mask_size * mask_size
        self.fc_similarity = nn.Linear(self.flat_input, self.flat_input)

        # Merge layer
        self.fc_merge = nn.Linear(
            512+self.word_embedding_size+self.flat_input, 512)

    def forward(self, inp):
        # x is the observation
        # y is the target
        # z is the object location mask
        (x, y, z) = inp

        x = x.view(-1)
        x = self.fc_observation(x)
        x = F.relu(x, True)

        y = y.view(-1)
        y = self.fc_target(y)
        y = F.relu(y, True)

        z = z.view(-1)
        z = self.fc_similarity(z)
        z = F.relu(z, True)

        # xy = torch.stack([x, y], 0).view(-1)
        xyz = torch.cat([x, y, z])
        xyz = self.fc_merge(xyz)
        xyz = F.relu(xyz, True)
        return xyz


class word2vec_notarget_resnet50(nn.Module):
    """Our method network without target word embedding (original baseline with resnet50 from context grid)
    """

    def save_gradient(self, grad):
        self.gradient = grad

    def hook_backward(self, module, grad_input, grad_output):
        self.gradient_vanilla = grad_input[0]

    def __init__(self, method, mask_size=5):
        super(word2vec_notarget, self).__init__()

        self.gradient = None
        self.gradient_vanilla = None
        self.conv_output = None
        self.output_context = None

        # Observation layer
        self.fc_observation = nn.Linear(8192, 512)

        # Convolution for similarity grid
        pooling_kernel = 2
        self.conv1 = nn.Conv2d(1, 8, 3, stride=1)
        self.conv1.register_backward_hook(self.hook_backward)
        self.pool = nn.MaxPool2d(pooling_kernel, pooling_kernel)
        self.conv2 = nn.Conv2d(8, 16, 5, stride=1)

        conv1_output = (mask_size - 3 + 1)//pooling_kernel
        conv2_output = (conv1_output - 5 + 1)//pooling_kernel
        self.flat_input = 16 * conv2_output * conv2_output

        # Merge layer
        self.fc_merge = nn.Linear(
            512+self.flat_input, 512)

    def forward(self, inp):
        # x is the observation
        # z is the object location mask
        (x, z) = inp
        x = x.view(-1,2048,4).mean(0).view(-1) # for spatial features

        # x = x.view(-1)
        x = self.fc_observation(x)
        x = F.relu(x, True)

        z = torch.autograd.Variable(z, requires_grad=True)
        z = self.conv1(z)
        z.register_hook(self.save_gradient)
        self.conv_output = z
        z = self.pool(F.relu(z))
        z = self.pool(F.relu(self.conv2(z)))
        z = z.view(-1)
        self.output_context = z

        # xy = torch.stack([x, y], 0).view(-1)
        xyz = torch.cat([x, z])
        xyz = self.fc_merge(xyz)
        xyz = F.relu(xyz, True)
        return xyz

class word2vec_notarget(nn.Module):
    """modified network without target word embedding using resnet18
    """

    def save_gradient(self, grad):
        self.gradient = grad

    def hook_backward(self, module, grad_input, grad_output):
        self.gradient_vanilla = grad_input[0]

    def __init__(self, method, mask_size=5):
        super(word2vec_notarget, self).__init__()

        self.gradient = None
        self.gradient_vanilla = None
        self.conv_output = None
        self.output_context = None

        # Observation layer
        self.fc_observation = nn.Linear(2048, 512)

        # Convolution for similarity grid
        pooling_kernel = 2
        self.conv1 = nn.Conv2d(1, 8, 3, stride=1)
        # self.conv1.register_backward_hook(self.hook_backward)
        self.pool = nn.MaxPool2d(pooling_kernel, pooling_kernel)
        self.conv2 = nn.Conv2d(8, 16, 5, stride=1)

        conv1_output = (mask_size - 3 + 1)//pooling_kernel
        conv2_output = (conv1_output - 5 + 1)//pooling_kernel
        self.flat_input = 16 * conv2_output * conv2_output

        # Merge layer
        self.fc_merge = nn.Linear(
            512+self.flat_input, 512)

    def forward(self, inp):
        # x is the observation
        # z is the object location mask
        (x, z) = inp

        x = x.view(512,49,-1).mean(1).view(-1)
        x = self.fc_observation(x)
        x = F.relu(x, True)

        z = torch.autograd.Variable(z, requires_grad=True)
        z = self.conv1(z)
        z.register_hook(self.save_gradient)
        self.conv_output = z
        z = self.pool(F.relu(z))
        z = self.pool(F.relu(self.conv2(z)))
        z = z.view(-1)
        self.output_context = z

        # xy = torch.stack([x, y], 0).view(-1)
        xyz = torch.cat([x, z])
        xyz = self.fc_merge(xyz)
        xyz = F.relu(xyz, True)
        return xyz

class ANA(nn.Module):
    """
    Our proposed attention-driven navigation agent
    """

    def save_gradient(self, grad):
        self.gradient = grad

    def hook_backward(self, module, grad_input, grad_output):
        self.gradient_vanilla = grad_input[0]

    def __init__(self, method, mask_size=5):
        super(ANA, self).__init__()

        self.gradient = None
        self.gradient_vanilla = None
        self.conv_output = None
        self.output_context = None

        # Observation layer
        self.att_conv_1 = nn.Conv2d(2048, 256, 3,stride=1,padding=1)
        self.att_conv_2 = nn.Conv2d(256, 128, 1)
        self.att_conv_3 = nn.Conv2d(144, 1, 1)
        self.grid_conv_1 = nn.Conv2d(1, 8, 3, stride=1,padding=1)
        self.grid_conv_2 = nn.Conv2d(8, 16, 5, stride=1)

        self.grid_pool = nn.AdaptiveMaxPool2d((7,7))

        self.att_conv_4 = nn.Conv2d(1, 8, 3, stride=1,padding=1)
        self.att_conv_5 = nn.Conv2d(8, 16, 3, stride=1,padding=1)
        self.flat_att = 16*7*7

        # Convolution for similarity grid
        pooling_kernel = 2
        self.conv1 = nn.Conv2d(1, 8, 3, stride=1)
        # self.conv1.register_backward_hook(self.hook_backward)
        self.pool = nn.MaxPool2d(pooling_kernel, pooling_kernel)
        self.conv2 = nn.Conv2d(8, 16, 5, stride=1)

        conv1_output = (mask_size - 3 + 1)//pooling_kernel
        conv2_output = (conv1_output - 5 + 1)//pooling_kernel
        self.flat_input = 16 * conv2_output * conv2_output

        # Merge layer
        self.fc_merge = nn.Linear(
            self.flat_att+self.flat_input, 512)

    def forward(self, inp):
        # x is the observation
        # z is the object location mask
        (x, z) = inp
        z = torch.autograd.Variable(z, requires_grad=True)

        x = x.transpose(0,1).contiguous() 
        x = x.view(-1,7,7).unsqueeze(0)
        att_x = F.relu(self.att_conv_1(x))
        att_x = F.relu(self.att_conv_2(att_x))

        att_z = self.grid_conv_1(z)
        att_z = F.relu(self.grid_conv_2(F.relu(att_z)))       
        att_z = self.grid_pool(att_z)

        att = torch.cat((att_x,att_z),dim=1)
        att = self.att_conv_3(att)
        att = F.softmax(att.view(1,1,-1),dim=-1)

        att = att.view(1,1,7,7)
        att_x = F.relu(self.att_conv_4(att))
        att_x = F.relu(self.att_conv_5(att_x))
        att_x = att_x.view(-1)

        z = self.conv1(z)
        z = self.pool(F.relu(z))
        z = self.pool(F.relu(self.conv2(z)))
        z = z.view(-1)
        
        xyz = torch.cat([att_x,z])
        xyz = self.fc_merge(xyz)
        xyz = F.relu(xyz, True)
        return (xyz,att.squeeze())


class word2vec_notarget_lstm_mask(nn.Module):
    """Our method network with LSTM without target word embedding 
    """

    def __init__(self, method, mask_size=5, nb_layer=1, cell="lstm"):
        super(word2vec_notarget_lstm, self).__init__()

        self.gradient = None
        self.gradient_vanilla = None
        self.conv_output = None
        self.output_context = None
        self.lstm_hidden = None
        self.cell = cell

        # Observation layer, use only last RGB frame
        self.fc_observation = nn.Linear(2048, 512)

        # Convolution for similarity grid
        pooling_kernel = 2
        self.conv1 = nn.Conv2d(1, 8, 3, stride=1)
        self.pool = nn.MaxPool2d(pooling_kernel, pooling_kernel)
        self.conv2 = nn.Conv2d(8, 16, 5, stride=1)

        conv1_output = (mask_size - 3 + 1)//pooling_kernel
        conv2_output = (conv1_output - 5 + 1)//pooling_kernel
        self.flat_input = 16 * conv2_output * conv2_output

        # Merge layer
        self.fc_merge = nn.Linear(
            512+self.flat_input, 512)

        # LSTM for merge layer
        if cell == "lstm":
            self.lstm = nn.LSTM(512, 512, num_layers=nb_layer)
        elif cell == 'rnn':
            self.lstm = nn.RNN(512, 512, num_layers=nb_layer)
        elif cell == 'gru':
            self.lstm = nn.GRU(512, 512, num_layers=nb_layer)

    def forward(self, inp):
        # x is the observation
        # z is the object location mask
        (x, z, hidden) = inp

        # x = x.view(-1)
        x = x.view(512,49,-1).mean(1).view(-1)
        x = self.fc_observation(x)
        x = F.relu(x, True)

        z = torch.autograd.Variable(z, requires_grad=True)
        z = self.conv1(z)
        z = self.pool(F.relu(z))
        z = self.pool(F.relu(self.conv2(z)))
        z = z.view(-1)
        self.output_context = z

        # xy = torch.stack([x, y], 0).view(-1)
        xyz = torch.cat([x, z])
        xyz = self.fc_merge(xyz)
        xyz = F.relu(xyz, True)
        if self.cell == "lstm":
            out, (h1, c1) =  self.lstm(xyz.view(1, 1, -1), hidden)
        else:
            out, h1 = self.lstm(xyz.view(1, 1, -1), hidden)

        return out.squeeze()

class word2vec_notarget_lstm(nn.Module):
    """Our method network with LSTM with target word embedding 
    """

    def __init__(self, method, mask_size=5, nb_layer=1, cell="lstm"):
        super(word2vec_notarget_lstm, self).__init__()

        self.gradient = None
        self.gradient_vanilla = None
        self.conv_output = None
        self.output_context = None
        self.lstm_hidden = None
        self.cell = cell
        self.word_embedding_size = 300


        # Observation layer, use only last RGB frame
        self.fc_observation = nn.Linear(512, 512)
        self.fc_target = nn.Linear(self.word_embedding_size, self.word_embedding_size)

        # Merge layer
        self.fc_merge = nn.Linear(
            512+self.word_embedding_size, 512)

        # LSTM for merge layer
        if cell == "lstm":
            self.lstm = nn.LSTM(512, 512, num_layers=nb_layer)
        elif cell == 'rnn':
            self.lstm = nn.RNN(512, 512, num_layers=nb_layer)
        elif cell == 'gru':
            self.lstm = nn.GRU(512, 512, num_layers=nb_layer)

    def forward(self, inp):
        # x is the observation
        # z is the object location mask
        (x, y, hidden) = inp

        # x = x.view(-1)
        x = x.view(512,49).mean(-1).view(-1)
        x = self.fc_observation(x)
        x = F.relu(x, True)

        y = y.view(-1)
        y = self.fc_target(y)
        y = F.relu(y, True)

        # xy = torch.stack([x, y], 0).view(-1)
        xyz = torch.cat([x, y])
        xyz = self.fc_merge(xyz)
        xyz = F.relu(xyz, True)
        if self.cell == "lstm":
            out, (h1, c1) =  self.lstm(xyz.view(1, 1, -1), hidden)
        else:
            out, h1 = self.lstm(xyz.view(1, 1, -1), hidden)

        return out.squeeze()

class baseline(nn.Module):
    """Baseline network
    """

    def __init__(self, method, mask_size=5):
        super(baseline, self).__init__()
        self.word_embedding_size = 300
        self.fc_target = nn.Linear(
            self.word_embedding_size, self.word_embedding_size)
        # Observation layer
        self.fc_observation = nn.Linear(8192, 512)
        self.fc_merge = nn.Linear(self.word_embedding_size + 512, 512)

        self.output_resnet = None

    def forward(self, inp):
        # x is the observation
            # y is the target
        (x, y) = inp

        x = x.view(-1)
        x = self.fc_observation(x)
        x = F.relu(x, True)
        self.output_resnet = x

        y = y.view(-1)
        y = self.fc_target(y)
        y = F.relu(y, True)

        xy = torch.cat([x, y])
        xy = self.fc_merge(xy)
        xy = F.relu(xy, True)
        return xy


class aop(nn.Module):
    """AOP with image feature as target
    """

    def __init__(self, method, mask_size=5):
        super(aop, self).__init__()
        # Target object layer
        self.fc_target = nn.Linear(2048, 512)

        # Observation layer
        self.fc_observation = nn.Linear(8192, 512)

        # Merge layer
        self.fc_merge = nn.Linear(1024+(mask_size*mask_size), 512)

    def forward(self, inp):
        # x is the observation
        # y is the target
        # z is the object location mask
        (x, y, z) = inp

        x = x.view(-1)
        x = self.fc_observation(x)
        x = F.relu(x, True)

        y = y.view(-1)
        y = self.fc_target(y)
        y = F.relu(y, True)

        z = z.view(-1)

        xy = torch.stack([x, y], 0).view(-1)
        xyz = torch.cat([xy, z])
        xyz = self.fc_merge(xyz)
        xyz = F.relu(xyz, True)
        return xyz


class aop_we(nn.Module):
    """AOP with word embedding as target
    """

    def __init__(self, method, mask_size=5):
        super(aop_we, self).__init__()
        # Target object layer
        self.fc_target = nn.Linear(300, 300)

        # Observation layer
        self.fc_observation = nn.Linear(8192, 512)

        # Merge layer
        self.fc_merge = nn.Linear(812+(mask_size*mask_size), 512)

    def forward(self, inp):
        # x is the observation
            # y is the target
            # z is the object location mask
        (x, y, z) = inp

        x = x.view(-1)
        x = self.fc_observation(x)
        x = F.relu(x, True)

        y = y.view(-1)
        y = self.fc_target(y)
        y = F.relu(y, True)

        z = z.view(-1)

        xyz = torch.cat([x, y, z])
        xyz = self.fc_merge(xyz)
        xyz = F.relu(xyz, True)
        return xyz


class target_driven(nn.Module):
    """Target driven using visual input as target
    """

    def __init__(self, method, mask_size=5):
        super(target_driven, self).__init__()
        # Siemense layer
        self.fc_siemense = nn.Linear(8192, 512)

        # Merge layer
        self.fc_merge = nn.Linear(1024, 512)

    def forward(self, inp):
        (x, y,) = inp

        x = x.view(-1)
        x = self.fc_siemense(x)
        x = F.relu(x, True)

        y = y.view(-1)
        y = self.fc_siemense(y)
        y = F.relu(y, True)

        xy = torch.stack([x, y], 0).view(-1)
        xy = self.fc_merge(xy)
        xy = F.relu(xy, True)
        return xy


class gcn(nn.Module):
    """GCN implementation
    """

    def __init__(self, method, mask_size=5):
        super(gcn, self).__init__()
        self.word_embedding_size = 300
        self.fc_target = nn.Linear(
            self.word_embedding_size, self.word_embedding_size)
        # Observation layer
        self.fc_observation = nn.Linear(8192, 512)

        # GCN layer
        self.gcn = GCN()

        # Merge word_embedding(300) + observation(512) + gcn(512)
        self.fc_merge = nn.Linear(
            self.word_embedding_size + 512 + 512, 512)

    def forward(self, inp):
        # x is the observation (resnet feature stacked)
        # y is the target
        # z is the observation (RGB frame)
        (x, y, z) = inp

        # x = x.view(-1)
        x = x.view(-1,2048,4).mean(0).view(-1) # for spatial features

        x = self.fc_observation(x)
        x = F.relu(x, True)

        y = y.view(-1)
        y = self.fc_target(y)
        y = F.relu(y, True)

        z = self.gcn(z)

        # xy = torch.stack([x, y], 0).view(-1)
        xyz = torch.cat([x, y, z])
        xyz = self.fc_merge(xyz)
        xyz = F.relu(xyz, True)
        return xyz


class SharedNetwork(nn.Module):
    """ Bottom network, will extract feature for the policy network
    """

    def __init__(self, method, mask_size=5):
        super(SharedNetwork, self).__init__()
        self.method = method
        self.gradient = None
        self.gradient_vanilla = None
        self.conv_output = None

        if self.method == 'ana':
            self.net = ANA(method, mask_size=mask_size)
        elif self.method == 'word2vec':
            self.net = word2vec(method, mask_size=mask_size)
        elif self.method == 'word2vec_noconv':
            self.net = word2vec_noconv(method, mask_size=mask_size)
        elif self.method == 'word2vec_notarget':
            self.net = word2vec_notarget(method, mask_size=mask_size)
        elif self.method == 'word2vec_notarget_lstm':
            self.net = word2vec_notarget_lstm(method, mask_size=mask_size)
        elif self.method == 'word2vec_notarget_lstm_2layer':
            self.net = word2vec_notarget_lstm(method, mask_size=mask_size, nb_layer=2)
        elif self.method == 'word2vec_notarget_lstm_3layer':
            self.net = word2vec_notarget_lstm(method, mask_size=mask_size, nb_layer=3)
        elif self.method == 'word2vec_notarget_rnn':
            self.net = word2vec_notarget_lstm(method, mask_size=mask_size, cell='rnn')
        elif self.method == 'word2vec_notarget_gru' :
            self.net = word2vec_notarget_lstm(method, mask_size=mask_size, cell='gru')

        # word2vec_nosimi is the baseline
        elif self.method == "word2vec_nosimi":
            self.net = baseline(method, mask_size=mask_size)
        elif self.method == 'aop':
            self.net = aop(method, mask_size=mask_size)
        elif self.method == 'aop_we':
            self.net = aop_we(method, mask_size=mask_size)
        elif self.method == 'target_driven':
            self.net = target_driven(method, mask_size=mask_size)
        elif self.method == 'gcn':
            self.net = gcn(method, mask_size=mask_size)
        else:
            raise Exception("Please choose a method")

    def save_gradient(self, grad):
        self.gradient = grad

    def hook_backward(self, module, grad_input, grad_output):
        self.gradient_vanilla = grad_input[0]

    def forward(self, inp):
        return self.net(inp)


class SceneSpecificNetwork(nn.Module):
    """
    Input for this network is 512 tensor (original)
    """

    def __init__(self, action_space_size):
        super(SceneSpecificNetwork, self).__init__()
        self.fc1 = nn.Linear(512, 512)

        # Policy layer
        self.fc2_policy = nn.Linear(512, action_space_size)

        # Value layer
        self.fc2_value = nn.Linear(512, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x_policy = self.fc2_policy(x)
        # x_policy = F.softmax(x_policy)

        x_value = self.fc2_value(x)[0]
        return (x_policy, x_value, )

# mask derived based on heuristics
# creating attention masks for 9 different actions
initial_mask = []
# move foward
MF_mask = torch.zeros(7,7)
MF_mask[2:4,3] = 1 # originally 2-5
initial_mask.append(MF_mask)
# rotate right
RR_mask = torch.zeros(7,7)
RR_mask[1:4,5] = 1 # originally 5
initial_mask.append(RR_mask)
# rotate left 
RL_mask = torch.zeros(7,7)
RL_mask[1:4,1] = 1 # originally 2
initial_mask.append(RL_mask)
# move back
MB_mask = torch.zeros(7,7)
MB_mask[4:6,3] = 1 # originally 2-5
initial_mask.append(MB_mask)
# look up
LU_mask = torch.zeros(7,7)
LU_mask[:2,2:5] = 1 # originally 2
initial_mask.append(LU_mask)
# look down
LD_mask = torch.zeros(7,7)
LD_mask[6:,2:5] = 1 # originally 5
initial_mask.append(LD_mask)
# move right
MR_mask = torch.zeros(7,7)
MR_mask[2:5,4] = 1 # originally 4
initial_mask.append(MR_mask)
# move left
ML_mask = torch.zeros(7,7)
ML_mask[2:5,2] = 1 # originally 3
initial_mask.append(ML_mask)
# DONE
done_mask = torch.zeros(7,7)
initial_mask.append(done_mask)
initial_mask = torch.cat([_.unsqueeze(0) for _ in initial_mask], dim=0)
initial_mask = initial_mask.view(-1,49)


class SceneSpecificNetwork_att2act(nn.Module):
    """
    Input for this network is 512 tensor (original)
    """

    def __init__(self, action_space_size):
        super(SceneSpecificNetwork_att2act, self).__init__()
        self.fc1 = nn.Linear(512, 512)

        # Policy layer
        self.fc2_policy = nn.Linear(512, action_space_size)

        # Value layer
        self.fc2_value = nn.Linear(512, 1)

        # att2act mask 
        self.mask = nn.Parameter(initial_mask,requires_grad=True) # initialize based on heuristics

        # weights for att2act
        self.att_weight = nn.Parameter(torch.zeros(1,),requires_grad=True) # trainable balance factors for all actions

    def forward(self, x):
        x, att = x
        x = self.fc1(x)
        x = F.relu(x)
        x_policy = self.fc2_policy(x)
        att2act = torch.mm(self.mask,att.view(49,1)).squeeze(-1)

        # adaptive balance factor
        att2act = torch.mul(att2act,torch.sigmoid(self.att_weight).expand_as(att2act)) # single factor


        x_policy = x_policy + att2act # residual learning

        x_value = self.fc2_value(x)[0]
        return (x_policy, x_value, ) # for training
        # return (x_policy, x_value, att) # for visualization


class ActorCriticLoss(nn.Module):
    def __init__(self, entropy_beta):
        self.entropy_beta = entropy_beta

    def forward(self, policy, value, action_taken, temporary_difference, r):
        # Calculate policy entropy
        log_softmax_policy = torch.nn.functional.log_softmax(policy, dim=1)
        softmax_policy = torch.nn.functional.softmax(policy, dim=1)
        policy_entropy = softmax_policy * log_softmax_policy
        policy_entropy = -torch.sum(policy_entropy, 1)

        # Policy loss
        nllLoss = F.nll_loss(log_softmax_policy, action_taken, reduce=False)
        policy_loss = nllLoss * temporary_difference - policy_entropy * self.entropy_beta
        policy_loss = policy_loss.sum(0)

        # Value loss
        # learning rate for critic is half of actor's
        # Equivalent to 0.5 * l2 loss
        value_loss = (0.5 * 0.5) * F.mse_loss(value, r, size_average=False)
        return value_loss + policy_loss


# Code borrowed from https://github.com/tkipf/pygcn
class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'


# Code borrowed from https://github.com/allenai/savn/
def normalize_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


class GCN(nn.Module):
    def __init__(self):
        super(GCN, self).__init__()

        self.resnet50 = models.resnet50(pretrained=True)
        for p in self.resnet50.parameters():
            p.requires_grad = False
        self.resnet50.eval()

        # Load adj matrix for GCN
        A_raw = torch.load("./data/gcn/adjmat.dat")
        A = normalize_adj(A_raw).tocsr().toarray()
        self.A = torch.nn.Parameter(torch.Tensor(A))

        objects = open("./data/gcn/objects.txt").readlines()
        objects = [o.strip() for o in objects]
        self.n = len(objects)
        self.register_buffer('all_glove', torch.zeros(self.n, 300))

        # Every dataset contain the same word embedding use FloorPlan1
        h5_file = h5py.File("./data/FloorPlan1.h5", 'r')
        object_ids = json.loads(h5_file.attrs['object_ids'])
        object_vector = h5_file['object_vector']

        word_embedding = {k: object_vector[v] for k, v in object_ids.items()}
        for i, o in enumerate(objects):
            self.all_glove[i, :] = torch.from_numpy(word_embedding[o])

        h5_file.close()

        nhid = 1024
        # Convert word embedding to input for gcn
        self.word_to_gcn = nn.Linear(300, 512)

        # Convert resnet feature to input for gcn
        self.resnet_to_gcn = nn.Linear(1000, 512)

        # GCN net
        self.gc1 = GraphConvolution(512 + 512, nhid)
        self.gc2 = GraphConvolution(nhid, nhid)
        self.gc3 = GraphConvolution(nhid, 1)

        self.mapping = nn.Linear(self.n, 512)

    def gcn_embed(self, x):

        resnet_score = self.resnet50(x)
        resnet_embed = self.resnet_to_gcn(resnet_score)
        word_embedding = self.word_to_gcn(self.all_glove)

        output = torch.cat(
            (resnet_embed.repeat(self.n, 1), word_embedding), dim=1)
        return output

    def forward(self, x):

        # x = (current_obs)
        # Convert input to gcn input
        x = self.gcn_embed(x)

        x = F.relu(self.gc1(x, self.A))
        x = F.relu(self.gc2(x, self.A))
        x = F.relu(self.gc3(x, self.A))
        x = x.view(-1)
        x = self.mapping(x)
        return x
