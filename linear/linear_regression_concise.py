'''
Author: QDX
Date: 2023-04-19 09:30:59
Description: 
'''
import os
import sys
import numpy as np
import torch
from torch import nn
from torch.utils import data
from d2l import torch as d2l

pwd = os.path.dirname(os.path.abspath(__file__))
base_path = os.path.dirname(pwd)
sys.path.append(base_path)

from struct_data_debug import debug_tensor


def load_array(data_arrays, batch_size, is_train=True):
    """构造一个PyTorch数据迭代器"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


def run_model():
    true_w = torch.tensor([2, -3.4])
    true_b = 4.2
    features, labels = d2l.synthetic_data(true_w, true_b, 1000)
    debug_tensor(features)

    batch_size = 10
    data_iter = load_array((features, labels), batch_size)
    print(next(iter(data_iter)))


    # 流水线 类 实例
    net = nn.Sequential(nn.Linear(2, 1))
    net[0].weight.data.normal_(0, 0.01)
    net[0].bias.data.fill_(0)
    # loss
    loss = nn.MSELoss()
    # 优化方法
    trainer = torch.optim.SGD(net.parameters(), lr=0.03)
    num_epochs = 3
    for epoch in range(num_epochs):
        for X, y in data_iter:
            l = loss(net(X), y)
            trainer.zero_grad()
            l.backward()
            trainer.step()
        l = loss(net(features), labels)
        print(f'epoch {epoch + 1}, loss {l:f}')

    w = net[0].weight.data
    print('w的估计误差：', true_w - w.reshape(true_w.shape))
    b = net[0].bias.data
    print('b的估计误差：', true_b - b)


if __name__ == "__main__":
    run_model()