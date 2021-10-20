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

#from convlstm import ConvLSTMCell

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
        self.sequences = [i for i in range(len(glob.glob(data_path + "/*")))]
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        arrays = [self.data_path + "/" + str(idx) + "/" + str(j) + ".png" for j in range(20)]
        images = np.array([ np.asarray(plt.imread(img_path))[:, :, :3]  for img_path in arrays])
        images = images[:20]
        #plt.imshow(images[0])
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
        temp = pixel_y
        pixel_y = pixel_x
        pixel_x = temp
        pixel_x = 165 - pixel_x
        
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

        #images = np.concatenate((np.zeros((20, 3, 165, 3)), images), axis=1)
        return torch.tensor(images.reshape(images.shape[0] * images.shape[3], images.shape[1],images.shape[2]), dtype=torch.float64), isotropicGrayscaleImage, images[0]

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
        heatmap = dec 
        #heatmap_norm = torch.linalg.matrix_norm(heatmap)
        #for i in range(len(heatmap)):
        #    heatmap[i] /= heatmap_norm[i]        
        #for i in range(len(heatmap)):
        x = heatmap.squeeze()
        x = nn.Softmax(2)(x.view(x.shape[0], 1, x.shape[1] * x.shape[2])).view_as(x)
        return x.squeeze()

class EnDeWithPooling(nn.Module):
    def __init__(self, activation, initType, numChannels, batchnorm=False, softmax=True):
        super(EnDeWithPooling, self).__init__()

        self.batchnorm = batchnorm
        self.bias = not batchnorm
        self.initType = initType
        self.activation = None
        self.numChannels = numChannels
        self.softmax = softmax

        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        else:
            self.activation = nn.SELU(inplace=True)

        self.conv1 = nn.Conv2d(self.numChannels, 16, 3, 1, 1, bias=self.bias)
        self.conv2 = nn.Conv2d(16, 32, 3, 1, 1, bias=self.bias)
        self.conv3 = nn.Conv2d(32, 64, 3, 1, 1, bias=self.bias)
        self.deconv3 = nn.ConvTranspose2d(64, 32, (4,4), 2, 1, 1)
        self.deconv2 = nn.ConvTranspose2d(32, 16, 3, 2, 1, 1)
        self.deconv1 = nn.ConvTranspose2d(16, 8, (3,4), 2, 1, 1)
        self.classifier = nn.Conv2d(8, 1, 1)

        self.pool = nn.MaxPool2d(2, 2)
        self.intermediate = nn.Conv2d(64, 64, 1, 1, 0, bias=self.bias)
        self.skip1 = nn.Conv2d(16, 16, 1, 1, 0, bias=self.bias)
        self.skip2 = nn.Conv2d(32, 32, 1, 1, 0, bias=self.bias)

        if self.batchnorm:
            self.bn1 = nn.BatchNorm2d(16)
            self.bn2 = nn.BatchNorm2d(32)
            self.bn3 = nn.BatchNorm2d(64)
            self.bn4 = nn.BatchNorm2d(32)
            self.bn5 = nn.BatchNorm2d(16)
            self.bn6 = nn.BatchNorm2d(8)

    def forward(self, x):
        if self.batchnorm:
            conv1_ = self.pool(self.bn1(self.activation(self.conv1(x))))
            conv2_ = self.pool(self.bn2(self.activation(self.conv2(conv1_))))
            conv3_ = self.pool(self.bn3(self.activation(self.conv3(conv2_))))
            intermediate_ = self.activation(self.intermediate(conv3_))
            skip_deconv3_ = self.deconv3(intermediate_) + self.activation(self.skip2(conv2_))
            deconv3_ = self.bn4(self.activation(skip_deconv3_))
            skip_deconv2_ = self.deconv2(deconv3_) + self.activation(self.skip1(conv1_))
            deconv2_ = self.bn5(self.activation(skip_deconv2_))
            deconv1_ = self.bn6(self.activation(self.deconv1(deconv2_)))
            score = self.classifier(deconv1_)
            score = score.squeeze()
            score = nn.Softmax(1)(score.view(score.shape[0], -1)).view_as(score)            
        else:
            conv1_ = self.pool(self.activation(self.conv1(x)))
            conv2_ = self.pool(self.activation(self.conv2(conv1_)))
            conv3_ = self.pool(self.activation(self.conv3(conv2_)))
            intermediate_ = self.activation(self.intermediate(conv3_))
            skip_deconv3_ = self.deconv3(intermediate_) + self.activation(self.skip2(conv2_))
            deconv3_ = self.activation(skip_deconv3_)
            skip_deconv2_ = self.deconv2(deconv3_) + self.activation(self.skip1(conv1_))
            deconv2_ = self.activation(skip_deconv2_)
            deconv1_ = self.activation(self.deconv1(deconv2_))
            if self.softmax:
                score = self.classifier(deconv1_)
            else:
                score = self.classifier(deconv1_)
        return score

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if self.initType == 'default':
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, np.sqrt(2. / n))
                elif self.initType == 'xavier':
                    nn.init.xavier_normal_(m.weight.data)

                if m.bias is not None:
                    m.bias.data.zero_()

            if isinstance(m, nn.ConvTranspose2d):
                if self.initType == 'default':
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, np.sqrt(2. / n))
                elif self.initType == 'xavier':
                    nn.init.xavier_normal_(m.weight.data)

                if m.bias is not None:
                    m.bias.data.zero_()

from tqdm import tqdm
if __name__ == "__main__":
    full_dataset = ArgoverseImageDataset(data_path="../results")
    train_size = int(1 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=False)
    #test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

    #model = EnDeWithPooling('relu','xavier',numChannels=60,batchnorm=True,softmax=True)
    model = HOME(in_s=(1, 60, 164, 165))
    model.double()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001)
    criterion = nn.KLDivLoss()
    #criterion = nn.MSELoss()
    #model.load_state_dict(torch.load('checkpoints/home_home.ckpt'))
    for epoch in range(100):
        train_loss = []
        for ind, data in enumerate(tqdm(train_loader)):
            inputs, gt, images = data
            #continue
            # print(gt[0,0].sum())
            out = model(inputs)
            loss = criterion(F.torch.log(out), gt)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step();
            train_loss.append(loss.item())
            print("loss=",loss.item())
            if False:
                for jj in range(len(images)):
                    plt.imshow(images[0])
                    plt.imshow(out[0].detach(), alpha=0.4)
                    plt.savefig('../intermediate_heatmap_results/{}_predicted.png'.format(jj + 4* ind))
                    plt.imshow(images[0])
                    plt.imshow(gt[0], alpha=0.4)
                    plt.savefig('../intermediate_heatmap_results/{}_gt.png'.format(jj + 4* ind))
            #plt.imshow(out[0].detach(),alpha=0.2)
            #plt.show()
            #print(out.shape, gt.shape)
        print("Min Loss=",np.min(np.array(train_loss)))
        print("Mean Loss = ", np.mean(np.array(train_loss)))
        if epoch % 2 != 0:
            continue
        '''
        for ind, data in enumerate(tqdm(test_loader)):
            inputs, gt, images = data
            #continue
            # print(gt[0,0].sum())
            print(inputs.shape)
            out = model(inputs)
            loss = criterion(F.torch.log(out), gt)
            #loss = criterion(out, gt)
            train_loss.append(loss.item())
            print("loss=",loss.item())
            if ind % 1 == 0:
                for jj in range(len(images)):
                    plt.imshow(images[0])
                    plt.imshow(out[0].detach(), alpha=0.4)
                    plt.savefig('../intermediate_heatmap_results/{}_predicted.png'.format(170+jj * ind))
                    plt.clf()
                    #plt.show()
                    plt.imshow(images[0])
                    plt.imshow(gt[0], alpha=0.4)
                    plt.savefig('../intermediate_heatmap_results/{}_gt.png'.format(170+jj * ind))
                    plt.clf()
                    #plt.show()
            #plt.imshow(out[0].detach(),alpha=0.2)
            #plt.show()
            #print(out.shape, gt.shape)
            print("Min Loss=",np.min(np.array(train_loss)))
            print("Mean Loss = ", np.mean(np.array(train_loss)))
        '''
        torch.save(model.state_dict(),'checkpoints/home_home.ckpt')
