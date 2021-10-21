import copy
from unicodedata import category
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

from convlstm import ConvLSTMCell

torch.autograd.set_detect_anomaly(True)
flag = True

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
    
class ArgoverseImageDataset(Dataset):
    def __init__(self, data_path, rep_path ="../inn", categories=['das', 'agents', 'others']):
        self.data_path = data_path
        self.sequences = [i for i in range(len(glob.glob(data_path + "/*")))]
        self.sequences = glob.glob(data_path + "/*")
        self.categories = categories
        self.rep_path = rep_path
    
    def __len__(self):
        return 10
    
    def __getitem__(self, idx):
        #arrays = [self.data_path + "/" + str(idx) + "/" + str(j) + ".png" for j in range(20)]
        #arrays = glob.glob(self.data_path + "/" + str(idx) + "/*")
        
        # dataset
        # import os
        gt = np.zeros((len(self.categories), 20, 512, 512))
        for cnt, i in enumerate(self.categories):
            gt[cnt] = np.load(self.rep_path + "/" + i + "/" + "{}.npy".format(idx))
        
        print(gt.shape)        
        
        # gt
        
        #plt.imshow(images[0])
        data_path="/datasets/argoverse/val/data"
        paths = glob.glob(os.path.join(data_path, "*.csv"))

        order = np.load("order.npy")
        path = data_path + "/" + order[idx].split('/')[-1]

        dff = pd.read_csv(path)
    
        city = dff['CITY_NAME'].values[0]    
        
        agent_df = dff[dff['OBJECT_TYPE'] == 'AGENT']
        x_a = agent_df['X'].values
        y_a = agent_df['Y'].values
        x_a, y_a = denoise(x_a, y_a)    

        pixel_x, pixel_y = 512 * (x_a[49] - x_a[20] + 50)/100, 512 * (y_a[49] - y_a[20] + 50)/100
        temp = pixel_y
        pixel_y = pixel_x
        pixel_x = temp
        pixel_x = 512 - pixel_x
        
        scaledGaussian = lambda x : exp(-(1/2)*(x**2))

        imgSize = 512
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
        x = isotropicGrayscaleImage.view(1, 1, 512, 512)
        x = nn.Softmax(2)(x.view(*x.size()[:2], -1)).view_as(x)
        x = x.view(512, 512)
        isotropicGrayscaleImage = x
        if flag:
            np.save("../gt_heatmaps/gt_{}.npy".format(idx), isotropicGrayscaleImage.detach().numpy())

        #images = np.concatenate((np.zeros((20, 3, 512, 3)), images), axis=1)
        return gt.reshape(60, 512, 512), isotropicGrayscaleImage

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
            temp = 1
            score = nn.Softmax(1)(score.view(score.shape[0], -1)/temp).view_as(score)            
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


class EnDeConvLSTM(nn.Module):
    def __init__(self, activation, initType, numChannels, imageHeight, imageWidth, batchnorm=False, softmax=False):
        super(EnDeConvLSTM, self).__init__()

        self.batchnorm = batchnorm
        self.bias = not self.batchnorm
        self.initType = initType
        self.activation = None
        self.batchsize = 1
        self.numChannels = numChannels
        self.softmax = softmax

        # Encoder
        self.conv1 = nn.Conv2d(self.numChannels, 16, 3, 1, 1, bias=self.bias)
        self.conv2 = nn.Conv2d(16, 32, 3, 1, 1, bias=self.bias)
        self.conv3 = nn.Conv2d(32, 64, 3, 1, 1, bias=self.bias)

        # Decoder
        self.deconv3 = nn.ConvTranspose2d(64, 32, 3, 2, 1, 1)
        self.deconv2 = nn.ConvTranspose2d(32, 16, 3, 2, 1, 1)
        self.deconv1 = nn.ConvTranspose2d(16, 8, 3, 2, 1, 1)

        # LSTM
        self.convlstm = ConvLSTMCell((int(imageWidth / 8), int(imageHeight / 8)), 64, 64, (3, 3))
        self.h, self.c = None, None

        # Skip Connections
        self.skip1 = nn.Conv2d(16, 16, 1, 1, 0, bias=self.bias)
        self.skip2 = nn.Conv2d(32, 32, 1, 1, 0, bias=self.bias)

        self.pool = nn.MaxPool2d(2, 2)
        self.classifier = nn.Conv2d(8, 1, 1)

        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        else:
            self.activation = nn.SELU(inplace=True)

        if self.batchnorm:
            self.bn1 = nn.BatchNorm2d(16)
            self.bn2 = nn.BatchNorm2d(32)
            self.bn3 = nn.BatchNorm2d(64)
            self.bn4 = nn.BatchNorm2d(32)
            self.bn5 = nn.BatchNorm2d(16)
            self.bn6 = nn.BatchNorm2d(8)

    def forward(self, x, s=None):
        if s is None:
            self.h, self.c = self.convlstm.init_hidden(self.batchsize)
        else:
            self.h, self.c = s

        if self.batchnorm is True:
            # Encoder
            conv1_ = self.pool(self.bn1(self.activation(self.conv1(x))))
            conv2_ = self.pool(self.bn2(self.activation(self.conv2(conv1_))))
            conv3_ = self.pool(self.bn3(self.activation(self.conv3(conv2_))))

            # LSTM
            self.h, self.c = self.convlstm(conv3_, (self.h, self.c))

            # Decoder
            deconv3_ = self.bn4(self.activation(self.deconv3(self.h)) + self.activation(self.skip2(conv2_)))
            deconv2_ = self.bn5(self.activation(self.deconv2(deconv3_)) + self.activation(self.skip1(conv1_)))
            deconv1_ = self.bn6(self.activation(self.deconv1(deconv2_)))

            if self.softmax:
                score = F.softmax(self.classifier(deconv1_), dim=1)
            else:
                score = self.classifier(deconv1_)
        else:
            # Encoder
            conv1_ = self.pool(self.activation(self.conv1(x)))
            conv2_ = self.pool(self.activation(self.conv2(conv1_)))
            conv3_ = self.pool(self.activation(self.conv3(conv2_)))

            # LSTM
            self.h, self.c = self.convlstm(conv3_, (self.h, self.c))

            # Decoder
            deconv3_ = self.activation(self.deconv3(self.h)) + self.activation(self.skip2(conv2_))
            deconv2_ = self.activation(self.deconv2(deconv3_)) + self.activation(self.skip1(conv1_))
            deconv1_ = self.activation(self.deconv1(deconv2_))
            if self.softmax:
                score = F.softmax(self.classifier(deconv1_), dim=1)
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
            if isinstance(m, ConvLSTMCell):
                n = m.conv.kernel_size[0] * m.conv.kernel_size[1] * m.conv.out_channels
                m.conv.weight.data.normal_(0, np.sqrt(2. / n))
                if m.conv.bias is not None:
                    m.conv.bias.data.zero_()



from tqdm import tqdm
if __name__ == "__main__":
    full_dataset = ArgoverseImageDataset(data_path="../results")
    train_size = int(1 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    #test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

    model = EnDeConvLSTM('relu','xavier',60,512,512,True, True)
    #model = HOME(in_s=(1, 60, 512, 512))
    model.double()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001)
    criterion = nn.KLDivLoss()
    #criterion = nn.MSELoss()
    #model.load_state_dict(torch.load('checkpoints/home_home.ckpt'))
    for epoch in range(100):
        train_loss = []
        for ind, data in enumerate(tqdm(train_loader)):
            inputs, gt = data
            #continue
            # print(gt[0,0].sum())
            out = model(inputs)
            break
            loss = criterion(F.torch.log(out), gt)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step();
            train_loss.append(loss.item())
            print("loss=",loss.item())
            if epoch % 2 == 0 and ind % 10 == 0:
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
        flag = False
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
        torch.save(model.state_dict(),'checkpoints/home_encdec.ckpt')
