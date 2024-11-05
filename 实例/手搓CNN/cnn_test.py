import torch
import torch.nn as nn#所有的层都包括在nn里、包括所有的激活函数
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms#包含图像增广和预处理的一系列操作
from torchvision import models
import torch.utils.data as Data#处理数据集的工具包,用于遍历数据
from torchvision.datasets import FashionMNIST#一个数据集 包含60000个数据
import torchsummary as summmary#展示模型参数

import numpy as np
import matplotlib.pyplot as plt
from cnn_prepare import LeNet
import time
import pandas as pd
import copy


def test_process():
    test_data = FashionMNIST(root='./data',
                             train=False,
                             # compose是一个容器类，在列表中放着一系列函数，用于封装打包一系列函数，对图像进行预处理和增广
                             transform=transforms.Compose([transforms.Resize(size=28), transforms.ToTensor()]),
                             download=True)
    test_dataloader = Data.DataLoader(dataset=test_data,
                                       batch_size=4,
                                       shuffle=True,
                                       num_workers=2)

    return test_dataloader

def model_test(model,test_dataloader):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print("gpu")
    else:
        print("cpu")
    model=model.to(device)

    test_correct=0
    test_num=0

    #只进行前向传播 不计算梯度，从而节省内存，加快运行速度
    with torch.no_grad():
        for b_x,b_y in test_dataloader:
            b_x = b_x.to(device)  # 图像数据
            b_y = b_y.to(device)  # 图像类别对应的序号/标签
            #验证模式
            model.eval()

            output=model(b_x)#每个类别的概率值
            pre_lab = torch.argmax(output, dim=1)#最大概率的label

            test_correct+=torch.sum(pre_lab==b_y.data)
            test_num += b_x.size(0)

    test_acc=test_correct/test_num
    print("test_acc=",test_acc)

if __name__=="__main__":
    #实例化
    model=LeNet()
    test_dataloader=test_process()
    model_test(model,test_dataloader)





