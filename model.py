import copy
import numpy as np

import os
import glob
import cv2

from PIL import Image

import torch
from torch import nn
import torch.nn.functional as F
from scipy.linalg import block_diag
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt

class ContextEncoder(nn.Module):
    def __init__(self, input_shape, batchnorm=False, numChannels=[32, 45, 64, 128, 256, 512], padding=1):
        super(ContextEncoder, self).__init__()
        self.batchnorm = batchnorm
        self.numChannels = numChannels  
        self.conv1 = nn.Conv2d(input_shape[1], numChannels[1] , kernel_size=(3,3), padding=(padding, padding))
        self.conv2 = nn.Conv2d(numChannels[1], numChannels[2] , kernel_size=(3,3), padding=(padding, padding))
        self.conv3 = nn.Conv2d(numChannels[2], numChannels[3] , kernel_size=(3,3), padding=(padding, padding))
        self.conv4 = nn.Conv2d(numChannels[3], numChannels[4] , kernel_size=(3,3), padding=(padding, padding))
        self.conv5 = nn.Conv2d(numChannels[4], numChannels[5] , kernel_size=(3,3), padding=(padding, padding))

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.activation = nn.ReLU()

    def forward(self, inputs):
        bn1 = self.pool(self.activation(self.conv1(inputs)))
        
        bn2 = self.pool(self.activation(self.conv2(bn1)))
        
        bn3 = self.pool(self.activation(self.conv3(bn2)))
        
        bn4 = self.pool(self.activation(self.conv4(bn3)))
        
        bn5 = self.pool(self.activation(self.conv5(bn4)))
        
        return bn5
        
class TemporalEncoder(nn.Module):
    def __init__(self):
        super(TemporalEncoder, self).__init__()
        self.conv = nn.Conv1d(4, 64, kernel_size=(3,3), padding=(2, 2))        
        self.activation = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.rnn = nn.GRU(input_size=64, hidden_size=128, batch_first= True)

    def forward(self, inputs):
        enc = self.activation(self.pool(self.conv(inputs)))
        out, _ = self.rnn(enc)
        return out

class TemporalEncoder(nn.Module):
    def __init__(self):
        super(TemporalEncoder, self).__init__()
        self.conv = nn.Transformer(128, )
        self.activation = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.rnn = nn.GRU(input_size=64, hidden_size=128, batch_first= True)

    def forward(self, inputs):
        enc = self.activation(self.pool(self.conv(inputs)))
        out, _ = self.rnn(enc)
        return out

class SocialEncoder(nn.Module):
    def __init__(self, query_dim, input_size):
        super(SocialEncoder, self).__init__()
        self.scale = 1.0/np.sqrt(query_dim)
        self.softmax = nn.Softmax(dim=2)
        self.layernorm = nn.LayerNorm(input_size)
        self.linear = nn.Linear(input_size, input_size)
        self.activation = nn.ReLU()
        

    def forward(self, enct, enco):
        query = enct
        keys, values = enco
        
        #! TODO
        
        # compute energy
        query = query.unsqueeze(1) # [B,Q] -> [B,1,Q]
        keys = keys.permute(1,2,0) # [T,B,K] -> [B,K,T]
        energy = torch.bmm(query, keys) # [B,1,Q]*[B,K,T] = [B,1,T]
        energy = self.softmax(energy.mul_(self.scale))
        
        # apply mask, renormalize
        # energy = energy*mask
        energy.div(energy.sum(2, keepdim=True))

        # weight values
        values = values.transpose(0,1) # [T,B,V] -> [B,T,V]
        combo = torch.bmm(energy, values).squeeze(1) # [B,1,T]*[B,T,V] -> [B,V]
        
        # pool context vector and enct
        pooled = combo, enct
        pooled = self.layernorm(pooled)
        
        # B x 1 x 128
        pooled = pooled.repeat([1, 14, 1])
        
        return pooled
    
class Decoder(nn.Module):
    def __init__(self, input_shape, batchnorm=False, numChannels=[256, 128, 64, 16, 1], padding=0):
        super(Decoder, self).__init__()
        self.batchnorm = batchnorm
        self.numChannels = numChannels  
        self.deconv1 = nn.ConvTranspose2d(input_shape[1], numChannels[0] , kernel_size=(3,3), padding=(padding, padding), stride=2)
        self.deconv2 = nn.ConvTranspose2d(numChannels[0], numChannels[1] , kernel_size=(3,3), padding=(padding, padding), stride=2)
        self.deconv3 = nn.ConvTranspose2d(numChannels[1], numChannels[2] , kernel_size=(3,3), padding=(padding, padding), stride=2)
        self.deconv4 = nn.ConvTranspose2d(numChannels[2], numChannels[3] , kernel_size=(3,3), padding=(padding, padding), stride=2)
        self.deconv5 = nn.ConvTranspose2d(numChannels[3], numChannels[4] , kernel_size=(3,3), padding=(padding, padding), stride=2)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
                
        self.activation = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
        self.MLP = nn.Linear(191*191, 164 * 165)

    def forward(self, inputs):
        bn1 = (self.activation(self.deconv1(inputs)))
        
        bn2 = (self.activation(self.deconv2(bn1)))
        
        bn3 = (self.activation(self.deconv3(bn2)))
        
        bn4 = (self.activation(self.deconv4(bn3)))
        
        bn5 = (self.activation(self.deconv5(bn4)))

        print(self.sigmoid(bn5).shape)

        return self.sigmoid(bn5)


class HOME(nn.Module):
    def __init__(self, in_s=(200, 200, 4), out_channels=512):
        super(HOME, self).__init__()
        self.ContextEncoder = ContextEncoder((in_s))
        # self.TemporalEncoder = TemporalEncoder()
        # self.SocialEncoder = SocialEncoder()
        # self.TransposeBlock1 = nn.ConvTranspose2d(640, 512,kernel_size=3,stride=1)
        # self.TransposeBlock2 = nn.ConvTranspose2d(512, 512,kernel_size=3,stride=1)
        self.Decoder = Decoder(input_shape=(1, 512, 5, 5))
        # self.activation = nn.ReLU()
    
    def forward(self, inputs):
        """forward

        Args:
            inputs (W x H x Nc): rasterized map with separate channels
            trajectory (N x Tobs x 3): scene context of agents
        """
        image_encodings = self.ContextEncoder(inputs)
        # temporal_agent = self.TemporalEncoder(trajectoryies[0])
        # temporal_others = self.TemporalEncoder(trajectoryies[1:])
        # social_encodings = self.SocialEncoder(temporal_agent, temporal_others)
        
        # common_encodings = torch.cat((image_encodings, social_encodings), dim=3) # of shape B, 14, 14, 512 and B, 14, 14, 128
        # common_encodings = self.activation(self.TransposeBlock1(common_encodings))
        # common_encodings = self.activation(self.TransposeBlock1(common_encodings))
        
        #heatmap = self.Decoder(image_encodings)
        print(image_encodings.shape)
        #print(heatmap.shape)
        return image_encodings
    
class ArgoverseImageDataset(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.sequences = glob.glob(data_path + "/*")
    
    def __len__(self):
        print(len(self.sequences))
        return len(self.sequences)
    
    def __getitem__(self, idx):
        arrays = glob.glob(self.sequences[idx] + "/*")
        images = np.array([ np.asarray(Image.open(img_path))  for img_path in arrays])
        images = images[:20]
        print(self.sequences[idx])
        return torch.tensor(images.reshape(images.shape[0] * images.shape[3], images.shape[1],images.shape[2]), dtype=torch.float64)

if __name__ == "__main__":
    train_dataset = ArgoverseImageDataset(data_path="../results")
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=False)
    model = HOME(in_s=(1, 80, 164, 165))
    model.double()
    for ind, data in enumerate(train_loader):
        model(data)
        break
