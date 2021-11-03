import copy
import pandas as pd
import numpy as np

import os
import glob
import cv2



from PIL import Image

import torch
from torch import nn
from math import exp
import torch.nn.functional as F
from scipy.linalg import block_diag
from torch.nn.modules import loss
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt

torch.autograd.set_detect_anomaly(True)

def denoise(gt_x, gt_y, w = 7):
    # denoising
    gt_x_t = []
    gt_y_t = []
    for iq in range(len(gt_x)):
        if iq >= w and iq + w <= len(gt_x):
            gt_x_t.append(np.mean(gt_x[iq: iq + w]))
            gt_y_t.append(np.mean(gt_y[iq: iq + w]))
        elif iq < w:
            okx = np.mean(gt_x[w: w + w])
            gt_x_t.append(gt_x[0] + (okx - gt_x[0]) * (iq) / w)
            oky = np.mean(gt_y[w: w + w])
            gt_y_t.append(gt_y[0] + (oky - gt_y[0]) * (iq) / w)
        else:
            okx = np.mean(gt_x[len(gt_x) - w:len(gt_x) - w  + w])
            oky = np.mean(gt_y[len(gt_x) - w: len(gt_x) - w + w])
            gt_x_t.append(okx + (gt_x[-1] - okx) * (w - (len(gt_x) - iq)) / w)
            gt_y_t.append(oky + (gt_y[-1] - oky) * (w - (len(gt_y) - iq)) / w)                   

    gt_x = gt_x_t
    gt_y = gt_y_t
    return gt_x, gt_y

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

        torch.nn.init.xavier_uniform(self.conv1.weight)
        torch.nn.init.xavier_uniform(self.conv2.weight)
        torch.nn.init.xavier_uniform(self.conv3.weight)
        torch.nn.init.xavier_uniform(self.conv4.weight)
        torch.nn.init.xavier_uniform(self.conv5.weight)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.activation = nn.ReLU()

    def forward(self, inputs):
        bn1 = self.pool(self.activation(self.conv1(inputs)))
        
        bn2 = self.pool(self.activation(self.conv2(bn1)))
        
        bn3 = self.pool(self.activation(self.conv3(bn2)))
        
        bn4 = self.pool(self.activation(self.conv4(bn3)))
        
        bn5 = self.pool(self.activation(self.conv5(bn4)))
        
        return bn5        
    
class Decoder(nn.Module):
    def __init__(self, input_shape, batchnorm=False, numChannels=[256, 128, 64, 16, 8, 1], padding=1):
        super(Decoder, self).__init__()
        self.batchnorm = batchnorm
        self.numChannels = numChannels  
        self.deconv1 = nn.ConvTranspose2d(input_shape[1], numChannels[0] , kernel_size=(3,3), padding=(padding + 1, padding + 1), stride=2, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(numChannels[0], numChannels[1] , kernel_size=(3,3), padding=(padding + 1, padding + 1), stride=2)
        self.deconv3 = nn.ConvTranspose2d(numChannels[1], numChannels[2] , kernel_size=(3,3), padding=(padding + 1, padding + 1), stride=2)
        self.deconv4 = nn.ConvTranspose2d(numChannels[2], numChannels[3] , kernel_size=(3,3), padding=(padding + 1, padding + 1), stride=2)
        self.deconv5 = nn.ConvTranspose2d(numChannels[3], numChannels[4] , kernel_size=(3,3), padding=(padding, padding), stride=2)
        self.deconv6 = nn.ConvTranspose2d(numChannels[4], numChannels[5] , kernel_size=(3,3), padding=(padding + 2, padding + 2), stride=2)
        self.deconv7 = nn.ConvTranspose2d(numChannels[5], numChannels[5] , kernel_size=(2,1), padding=(1, 0), stride=1)

        
        torch.nn.init.xavier_uniform(self.deconv1.weight)
        torch.nn.init.xavier_uniform(self.deconv2.weight)
        torch.nn.init.xavier_uniform(self.deconv3.weight)
        torch.nn.init.xavier_uniform(self.deconv4.weight)
        torch.nn.init.xavier_uniform(self.deconv5.weight)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
                
        self.activation = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        bn1 = (self.activation(self.deconv1(inputs)))
        
        bn2 = (self.activation(self.deconv2(bn1)))
        
        bn3 = (self.activation(self.deconv3(bn2)))
        
        bn4 = (self.activation(self.deconv4(bn3)))
        
        bn5 = (self.activation(self.deconv5(bn4)))

        bn6 = (self.activation(self.deconv6(bn5)))
        
        bn7 = (self.activation(self.deconv7(bn6)))

        return self.sigmoid(bn7)

class HOME2(nn.Module):
    def __init__(self, in_s=(200, 200, 4), out_channels=512):
        super(HOME2, self).__init__()
        self.ContextEncoder = ContextEncoder((in_s))
        self.Decoder = Decoder(input_shape=(1, 512, 5, 5))
    
    def forward(self, inputs):
        """forward

        Args:
            inputs (W x H x Nc): rasterized map with separate channels
            trajectory (N x Tobs x 3): scene context of agents
        """
        image_encodings = self.ContextEncoder(inputs)        
        heatmap = self.Decoder(image_encodings)
        heatmap = heatmap.squeeze()
        heatmap_norm = torch.linalg.matrix_norm(heatmap)
        for i in range(len(heatmap)):
            heatmap[i] /= heatmap_norm[i]
        return heatmap
    
class ArgoverseImageDataset(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.sequences = glob.glob(data_path + "/*")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        arrays = glob.glob(self.sequences[idx] + "/*")
        images = np.array([ np.asarray(Image.open(img_path))[:, :, :3]  for img_path in arrays])
        images = images[:20]
        if images.shape[1] != 164:
            print(self.sequences[idx] + "/*")
        data_path="/datasets/argoverse/val/data"
        paths = glob.glob(os.path.join(data_path, "*.csv"))
        path = paths[idx]
        dff = pd.read_csv(path)
    
        city = dff['CITY_NAME'].values[0]    
        
        agent_df = dff[dff['OBJECT_TYPE'] == 'AGENT']
        x_a = agent_df['X'].values
        y_a = agent_df['Y'].values    
        x_a, y_a = denoise(x_a, y_a)    

        pixel_x, pixel_y = 164 * (x_a[49] - x_a[20] + 50)/100, 165 * (y_a[49] - y_a[20] + 50)/100
        
        scaledGaussian = lambda x : exp(-(1/2)*(x**2))

        imgSize = 164
        isotropicGrayscaleImage = np.zeros((imgSize,imgSize + 1),np.uint8)

        for i in range(imgSize):
            for j in range(imgSize + 1):
                # find euclidian distance from center of image (imgSize/2,imgSize/2) 
                # and scale it to range of 0 to 2.5 as scaled Gaussian
                # returns highest probability for x=0 and approximately
                # zero probability for x > 2.5
                distanceFromCenter = np.linalg.norm(np.array([i-pixel_x,j-pixel_y]))
                distanceFromCenter = 2.5*distanceFromCenter/(imgSize/2)
                scaledGaussianProb = scaledGaussian(distanceFromCenter)
                isotropicGrayscaleImage[i,j] = np.clip(scaledGaussianProb*255,0,255)
        
        isotropicGrayscaleImage = torch.tensor(isotropicGrayscaleImage, dtype=torch.float64)
        x = isotropicGrayscaleImage.view(1, 1, 164, 165)
        x = nn.Softmax(2)(x.view(*x.size()[:2], -1)).view_as(x)
        x = x.view(164, 165)
        isotropicGrayscaleImage = x
        return torch.tensor(images.reshape(images.shape[0] * images.shape[3], images.shape[1],images.shape[2]), dtype=torch.float64), isotropicGrayscaleImage

class HOME(nn.Module):
    def __init__(self, in_s=(200, 200, 4), out_channels=512,):
        super(HOME, self).__init__()
        
        self.conv1 = nn.Conv2d(60, 64 , kernel_size=(3,3), padding=(1, 1))
        self.conv2 = nn.Conv2d(64, 128 , kernel_size=(3,3), padding=(1, 1))
        
        self.deconv1 = nn.ConvTranspose2d(128, 64 , kernel_size=(3,4), padding=(1, 1), stride=2, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(64, 32 , kernel_size=(1,1), padding=(0,0), stride=1)
        self.deconv3 = nn.ConvTranspose2d(32, 1 , kernel_size=(1,1), padding=(0,0), stride=1)
        batchNorm_momentum = 0.1
        
        self.bn11 = nn.BatchNorm2d(64, momentum=batchNorm_momentum)
        self.bn21 = nn.BatchNorm2d(128, momentum=batchNorm_momentum)
        self.bn31 = nn.BatchNorm2d(32, momentum=batchNorm_momentum)
        
        torch.nn.init.xavier_uniform(self.conv1.weight)
        torch.nn.init.xavier_uniform(self.conv2.weight)
        torch.nn.init.xavier_uniform(self.deconv1.weight)
        torch.nn.init.xavier_uniform(self.deconv2.weight)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.activation = nn.ReLU()

    def forward(self, inputs):
        enc =  self.pool(self.conv2(self.bn11(self.conv1(inputs))))
        dec =  self.activation(self.deconv3(self.bn31(self.deconv2(self.bn11(self.deconv1(enc))))))
        heatmap = nn.functional.softmax(dec.squeeze())
        #heatmap_norm = torch.linalg.matrix_norm(heatmap)
        #for i in range(len(heatmap)):
        #    heatmap[i] /= heatmap_norm[i]        
        #for i in range(len(heatmap)):
        x = heatmap
        x = x.view(x.shape[0], 1, x.shape[1], x.shape[2])
        x = nn.Softmax(2)(x.view(*x.size()[:2], -1)).view_as(x)
        return x.squeeze()

from tqdm import tqdm
if __name__ == "__main__":
    train_dataset = ArgoverseImageDataset(data_path="../results")
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=False)
    model = HOME(in_s=(1, 60, 164, 165))
    model.double()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
    criterion = nn.KLDivLoss()
    model.load_state_dict(torch.load('checkpoints/home.ckpt'))
    #criterion = nn.MSELoss()
    for epoch in range(100):
        train_loss = []
        for ind, data in enumerate(tqdm(train_loader)):
            inputs, gt = data
            # print(gt[0,0].sum())
            out = model(inputs)
            loss = criterion(F.torch.log(out), gt)
            #loss = criterion(out, gt)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step();
            train_loss.append(loss.item())
            print("loss=",loss.item())
            '''
            plt.imshow(gt[0])
            plt.show()
            plt.imshow(out[0].detach(),alpha=0.7)
            plt.show()
            print(out.shape, gt.shape)
            '''
        if epoch % 1 == 0:
            print("Min Loss=",np.min(np.array(train_loss)))
            print("Mean Loss = ", np.mean(np.array(train_loss)))
        
        torch.save(model.state_dict(),'checkpoints/home.ckpt')
