# Code adapted from: https://colab.research.google.com/github/abhimishra91/transformers-tutorials/blob/master/transformers_multiclass_classification.ipynb

#import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.optim as optim
from tqdm import tqdm

import numpy as np
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
from torch import cuda
import sys
import os
import csv
from sklearn.metrics import f1_score
import transformers
from transformers import AutoTokenizer, AutoModel
from nltk.corpus import stopwords

from torch.nn.parallel import DistributedDataParallel as DDP

LMTokenizer = AutoTokenizer.from_pretrained(sys.argv[1])
LMModel = AutoModel.from_pretrained(sys.argv[1])

device = 'cuda' if cuda.is_available() else 'cpu'
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#train_dataset = pd.read_csv('/scratch/amazon/dataset/train_chota_chota.csv', escapechar='\\', quoting=csv.QUOTE_NONE , names=['TITLE','DESCRIPTION','BULLET_POINTS','BRAND','BROWSE_NODE_ID'])
#testing_dataset = pd.read_csv('/scratch/amazon/dataset/test_chota_chota.csv', escapechar='\\', quoting=csv.QUOTE_NONE , names=['TITLE','DESCRIPTION','BULLET_POINTS','BRAND','BROWSE_NODE_ID'])

#train_dataset = pd.read_csv('/scratch/amazon/dataset/train_chota.csv', escapechar='\\', quoting=csv.QUOTE_NONE , names=['TITLE','DESCRIPTION','BULLET_POINTS','BRAND','BROWSE_NODE_ID'])
#testing_dataset = pd.read_csv('/scratch/amazon/dataset/test_chota.csv', escapechar='\\', quoting=csv.QUOTE_NONE , names=['TITLE','DESCRIPTION','BULLET_POINTS','BRAND','BROWSE_NODE_ID'])

train_dataset_w = pd.read_csv('/scratch/amazon2/dataset/train.csv', escapechar='\\', quoting=csv.QUOTE_NONE , names=['TITLE','DESCRIPTION','BULLET_POINTS','BRAND','BROWSE_NODE_ID'])
#train_dataset = pd.read_csv('/scratch/amazon2/dataset/train.csv', escapechar='\\', quoting=csv.QUOTE_NONE , names=['TITLE','DESCRIPTION','BULLET_POINTS','BRAND','BROWSE_NODE_ID'])
train_dataset = pd.read_csv('/scratch/amazon2/dataset/valid_top_1500.csv',  names=['TITLE','DESCRIPTION','BULLET_POINTS','BRAND','BROWSE_NODE_ID'])
testing_dataset = pd.read_csv('/scratch/amazon2/dataset/test.csv', escapechar='\\', quoting=csv.QUOTE_NONE , names=['PRODUCT_ID','TITLE','DESCRIPTION','BULLET_POINTS','BRAND'])

# label as per index on this array
uniques = train_dataset_w['BROWSE_NODE_ID'].value_counts().index.to_numpy()
dic = {}
for i in range(len(uniques)): dic[uniques[i]] = i
torch.cuda.empty_cache()
#print(train_dataset)

MAX_LEN = 512
TRAIN_BATCH_SIZE = 1 #int(sys.argv[2])
VALID_BATCH_SIZE = 1 #int(sys.argv[2])
LEARNING_RATE = 0.00002 #float(sys.argv[3])
drop_out = 0.3 #float(sys.argv[4])
EPOCHS = 4
#LABELS = 8
LABELS = len(uniques)
tokenizer = LMTokenizer

output_file_name = "anna.txt"
file = open(output_file_name, "w")

class Triage(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __getitem__(self, index):
        BRAND = str(self.data.BRAND[index])
        TITLE = str(self.data.TITLE[index]) + " " + BRAND
        TITLE = " ".join(TITLE.split())
        inputs = self.tokenizer.encode_plus(
            TITLE,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True,
            truncation=True
        )
        TITLE_ids = inputs['input_ids']
        TITLE_mask = inputs['attention_mask']

        BULLET_POINTS = str(self.data.BULLET_POINTS[index])
        DESCRIPTION = str(self.data.DESCRIPTION[index]) + BULLET_POINTS
        DESCRIPTION = " ".join(DESCRIPTION.split())
        inputs = self.tokenizer.encode_plus(
            DESCRIPTION,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True,
            truncation=True
        )
        DESCRIPTION_ids = inputs['input_ids']
        DESCRIPTION_mask = inputs['attention_mask']

        '''
        BULLET_POINTS = " ".join(BULLET_POINTS.split())
        inputs = self.tokenizer.encode_plus(
            BULLET_POINTS,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True,
            truncation=True
        )
        BULLET_POINTS_ids = inputs['input_ids']
        BULLET_POINTS_mask = inputs['attention_mask']

        '''

        '''return {
        return {
            'TITLE_ids': torch.tensor(TITLE_ids, dtype=torch.long).to(device, dtype = torch.long),
            'TITLE_mask': torch.tensor(TITLE_mask, dtype=torch.long).to(device, dtype = torch.long),
            
            'DESCRIPTION_ids': torch.tensor(DESCRIPTION_ids, dtype=torch.long).to(device, dtype = torch.long),
            'DESCRIPTION_mask': torch.tensor(DESCRIPTION_mask, dtype=torch.long).to(device, dtype = torch.long),
            
            'BULLET_POINTS_ids': torch.tensor(BULLET_POINTS_ids, dtype=torch.long).to(device, dtype = torch.long),
            'BULLET_POINTS_mask': torch.tensor(BULLET_POINTS_mask, dtype=torch.long).to(device, dtype = torch.long),
            
            'BRAND_ids': torch.tensor(BRAND_ids, dtype=torch.long).to(device, dtype = torch.long),
            'BRAND_mask': torch.tensor(BRAND_mask, dtype=torch.long).to(device, dtype = torch.long),
            
            'final_ids': torch.tensor(inputs['input_ids'], dtype=torch.long).to(device, dtype = torch.long),
            'final_mask': torch.tensor(inputs['attention_mask'], dtype=torch.long).to(device, dtype = torch.long),
            
        }'''
        return {
            'TITLE_ids': torch.tensor(TITLE_ids, dtype=torch.long),#.to(device, dtype = torch.long),
            'TITLE_mask': torch.tensor(TITLE_mask, dtype=torch.long),#.to(device, dtype = torch.long),
            
            'DESCRIPTION_ids': torch.tensor(DESCRIPTION_ids, dtype=torch.long),#.to(device, dtype = torch.long),
            'DESCRIPTION_mask': torch.tensor(DESCRIPTION_mask, dtype=torch.long),#.to(device, dtype = torch.long),
            
            #'BULLET_POINTS_ids': torch.tensor(BULLET_POINTS_ids, dtype=torch.long),#.to(device, dtype = torch.long),
            #'BULLET_POINTS_mask': torch.tensor(BULLET_POINTS_mask, dtype=torch.long),#.to(device, dtype = torch.long),
            
            #'BRAND_ids': torch.tensor(BRAND_ids, dtype=torch.long),#.to(device, dtype = torch.long),
            #'BRAND_mask': torch.tensor(BRAND_mask, dtype=torch.long),#.to(device, dtype = torch.long),
            
            'final_ids': torch.tensor(inputs['input_ids'], dtype=torch.long),#.to(device, dtype = torch.long),
            'final_mask': torch.tensor(inputs['attention_mask'], dtype=torch.long),#.to(device, dtype = torch.long),
            
            'BROWSE_NODE_ID': torch.tensor(np.where(uniques == int(self.data.BROWSE_NODE_ID[index]))[0][0], dtype=torch.long)
        } 
        #}''' 
    
    def __len__(self):
        return self.len

class LMClass(torch.nn.Module):
    def __init__(self):
        super(LMClass, self).__init__()
        self.l1 = LMModel
        self.l2 = LMModel
        self.pre_classifier = torch.nn.Linear(768 * 2, 768)
        self.dropout = torch.nn.Dropout(drop_out)
        self.classifier = torch.nn.Linear(768, LABELS)

    def forward(self, data):
        
        title_ids = data['TITLE_ids'].to(device, dtype = torch.long)
        title_mask = data['TITLE_mask'].to(device, dtype = torch.long)
        
        description_ids = data['DESCRIPTION_ids'].to(device, dtype = torch.long)
        description_mask = data['DESCRIPTION_mask'].to(device, dtype = torch.long)

        output_1 = self.l1(input_ids=title_ids, attention_mask=title_mask)
        hidden_state1 = output_1[0]
        pooler_1 = hidden_state1[:, 0]

        output_3 = self.l2(input_ids=description_ids, attention_mask=description_mask)
        hidden_state3 = output_3[0]
        pooler_3 = hidden_state3[:, 0]

        pooler = torch.cat((pooler_1, pooler_3), dim=1)
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.ReLU()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        return output

def calcuate_accu(big_idx, targets):
    n_correct = (big_idx==targets).sum().item()
    return n_correct

def train(epoch):
    tr_loss = 0
    n_correct = 0
    nb_tr_steps = 0
    nb_tr_examples = 0
    model.train()
    for _,data in enumerate(tqdm(training_loader)):
        targets = data['BROWSE_NODE_ID'].to(device, dtype = torch.long)
        outputs = model(data)
        loss = loss_function(outputs, targets)
        tr_loss += loss.item()
        big_val, big_idx = torch.max(outputs.data, dim=1)
        n_correct += calcuate_accu(big_idx, targets)

        nb_tr_steps += 1
        nb_tr_examples+=targets.size(0)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    file = open(output_file_name, "a");file.write(f'The Total Accuracy for Epoch {epoch}: {(n_correct*100)/nb_tr_examples}\n')
    epoch_loss = tr_loss/nb_tr_steps
    epoch_accu = (n_correct*100)/nb_tr_examples
    file.write(f"Training Loss Epoch: {epoch_loss}\n")
    file.write(f"Training Accuracy Epoch: {epoch_accu}\n")
    file.write("\n")
    file.close()
    return

training_set = Triage(train_dataset, tokenizer, MAX_LEN)
testing_set = Triage(testing_dataset, tokenizer, MAX_LEN)

train_params = {'batch_size': TRAIN_BATCH_SIZE,
                'shuffle': False,
                'pin_memory': True,
                'num_workers': 10
                }

test_params = {'batch_size': VALID_BATCH_SIZE,
                'shuffle': False,
                'pin_memory': True,
                'num_workers': 10
                }

training_loader = DataLoader(training_set, **train_params)
testing_loader = DataLoader(testing_set, **test_params)

model = LMClass()
# dist.init_process_group(backend='nccl', init_method='env://')
#model = nn.DataParallel(model)
model.load_state_dict(torch.load("/scratch/amazon2/dataset/final_big_.ckpt"))
model.to(device)
loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params = model.parameters(), lr=LEARNING_RATE)
gil = open("./epoch.txt", "w")
gil.close()

def inference(model):
    model.eval()
    to_write = []
    ind = 0
    '''ind = 0

    for i in tqdm(range(len(train_dataset['TITLE'].values))):
        TITLE = str(testing_dataset.TITLE.values[i])
        TITLE = " ".join(TITLE.split())
        inputs = LMTokenizer.encode_plus(
            TITLE,
            None,
            add_special_tokens=True,
            max_length=MAX_LEN,
            pad_to_max_length=True,
            return_token_type_ids=True,
            truncation=True
        )
        TITLE_ids = inputs['input_ids']
        TITLE_mask = inputs['attention_mask']
        print(torch.tensor(TITLE_ids, dtype=torch.long).shape)#TITLE_mask = inputs['attention_mask']
        data = {"final_ids": torch.tensor(TITLE_ids, dtype=torch.long), "final_mask": torch.tensor(TITLE_mask, dtype=torch.long)}#outputs = model(data) # of shape (16, 9919)
        outputs = model(data) # of shape (16, 9919)
        print(outputs.shape)#= model(data) # of shape (16, 9919)
    '''
    arr = []#for _, data in enumerate(tqdm(testing_loader)):
    for _, data in enumerate((training_loader)):
        outputs = model(data) # of shape (16, 9919)
        if torch.argmax(outputs[0]).item() !=  data["BROWSE_NODE_ID"][0].item(): print(torch.argmax(outputs[0]).item()  , dic[str(data["BROWSE_NODE_ID"][0].item())]);
        arr.extend([ [_ * TRAIN_BATCH_SIZE + i + 1, uniques[torch.argmax(outputs[i]).item()] ] for i in range((outputs.shape[0]))  ])#print(outputs.shape)#outputs = model(data) # of shape (16, 9919)
    #'''

    np.save("pred.npy", np.array(arr))
    file = open('submit.csv','w')
    cw = csv.writer(file)
    cw.writerows(arr)
    file.close()

inference(model);
