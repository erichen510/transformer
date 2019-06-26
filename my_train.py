import argparse
import math
import time

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from transformer.model import generate_model

num_epochs = 100

def pre_dataloader():
    train_loader = torch.utils.data.DataLoader()
    val_loader = torch.utils.data.DataLoader()
    return train_loader, val_loader


def main():

    device = torch.device('cuda' if  torch.cuda.is_available() else 'cpu')

    #===============  Loading Data ======================#

    data = torch.load() #todo:找个数据源利用BERT输出词向量或者字向量

    train_data, val_data  = pre_dataloader(data)

    n_src_vocab = 10000
    len_max_seq = 100

    final_model = generate_model(n_src_vocab, len_max_seq).to(device)

    train(final_model, train_data, val_data, device)


def train_epoch(model, train_data, optimizer, criterion, device):

    model.train()

    #train
    for batch in tqdm(
        train_data, mininterval=2, desc='- Tranning', leave=False):

        #提取数据
        src_seq, src_pos, label_bi, label_class = map(lambda x: x.to(device), batch)

        #forward
        optimizer.zero_grad()

        predict_left, predict_right =  model(src_seq, src_pos)
        loss_bi = criterion(predict_left, label_bi)
        loss_class = criterion(predict_right, label_class)

        loss = loss_bi + loss_class
        loss.backward()
        optimizer.step()

        #forward
        optimizer.zero_grad()



    return loss

def eval_epoch(model, val_data, optimizer, criterion, device):
    model.train()

    #train
    for batch in tqdm(
        val_data , mininterval=2, desc='- Validation', leave=False):

        #提取数据
        src_seq, src_pos, label_bi, label_class = map(lambda x: x.to(device), batch)

        #forward
        optimizer.zero_grad()

        predict_left, predict_right =  model(src_seq, src_pos)
        loss_bi = criterion(predict_left, label_bi)
        loss_class = criterion(predict_right, label_class)

        loss = loss_bi + loss_class

        return loss

def train(model, train_data, val_data, device):


    #确定梯度下降优化函数
    optimizer = optim.Adam(
        filter(lambda x: x.requires_grad, model.parameters()),
        betas = (0.9,0.98), eps = 1e-09)

    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        print('[Epoch', epoch, ']')

        start = time.time()
        train_loss = train_epoch(
            model, train_data, optimizer, criterion, device
        )

        start = time.time()
        val_loss = eval_epoch(model, val_data, optimizer, criterion, device)

        model_state_dict = model.state_dict()
        checkpoint = {
            'model': model_state_dict,
            'epoch': epoch
        }

        model_name = '-epoch{}.chkpt'.format(epoch+1)
        torch.save(checkpoint, model_name)


if __name__ == '__main__':
    main()
