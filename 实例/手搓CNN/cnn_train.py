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


#数据加载
def train_val_process():
    train_data = FashionMNIST(root='./data',
                              train=True,
                              # compose是一个容器类，在列表中放着一系列函数，用于封装打包一系列函数，对图像进行预处理和增广
                              transform=transforms.Compose([transforms.Resize(size=28), transforms.ToTensor()]),
                              download=True)

    #划分
    train_data,val_data=Data.random_split(train_data,[round(0.8*len(train_data)),round(0.2*len(train_data))])

    #载入
    train_dataloader = Data.DataLoader(dataset=train_data,
                                       batch_size=32,
                                       shuffle=True,  # 是否打乱数据
                                       num_workers=4)
    val_dataloader = Data.DataLoader(dataset=val_data,
                                     batch_size=32,
                                     shuffle=True,  # 是否打乱数据
                                     num_workers=4)

    return train_dataloader,val_dataloader


#完整训练过程
def model_train(model,train_dataloader,val_dataloader,num_epoch=100):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print("gpu")
    else:
        print("cpu")


    #优化器：用梯度算法来更新数据   其中就有Adam，SGB等优化过的梯度下降算法
    optimizer=optim.Adam(model.parameters(),lr=0.01) #model.parameters()储存的是权重和偏置

    #定义损失函数
    criterion=nn.CrossEntropyLoss()#在分类任务中，一般用交叉熵损失函数

    #把模型放入训练设备
    model=model.to(device)

    #最优模型初始化
    best_model_wts=copy.deepcopy(model.state_dict())#state_dict状态字典:用于保存一系列参数

    #参数初始化
    best_acc=0.0
    #损失储存
    train_loss_all=[]
    val_loss_all = []
    #准确度储存
    train_acc_all = []
    val_acc_all = []
    #时间
    since=time.time()

    #反向传播
    for epoch in range(num_epoch):
        print(epoch+1,'/',num_epoch)

        #初始化参数
        train_loss=0.0
        val_loss=0.0
        #预测正确的样本个数
        train_correct=0
        val_correct=0
        #样本数量
        train_num=0
        val_num=0

        #训练
        for step,(b_x,b_y) in enumerate(train_dataloader):#enumerate是枚举函数，返回（计数n，train_dataloader[n]）
            #一个batch_size大小的数据
            b_x=b_x.to(device)#图像数据img
            b_y=b_y.to(device)#图像类别对应的序号/标签lable
            #开启训练模式
            model.train()

            output=model(b_x)#通过类对象调用forward方法
            #找出最大概率的label的下标
            pre_lab=torch.argmax(output,dim=1)
            #计算损失
            loss=criterion(output,b_y)#这个loss计算的是一个batch中每个样本的平均loss
            #将梯度初始化为0
            optimizer.zero_grad()
            #反向传播计算/计算每个参数的梯度
            loss.backward()
            #通过梯度信息 对参数进行更新
            optimizer.step()

            train_loss+=loss.item()*b_x.size(0)#item()从张量中取出它的值
            train_correct+=torch.sum(pre_lab==b_y.data)
            train_num+=b_x.size(0)

        #验证:本质上是一个前向传播的过程，它不做反向传播
        for step,(b_x,b_y) in enumerate(val_dataloader):
            b_x = b_x.to(device)  # 图像数据
            b_y = b_y.to(device)  # 图像类别对应的序号/标签
            #验证模式
            model.eval()

            output=model(b_x)
            pre_lab = torch.argmax(output, dim=1)
            # 计算损失
            loss = criterion(output, b_y)

            val_loss += loss.item() * b_x.size(0)  # item()从张量中取出它的值
            val_correct += torch.sum(pre_lab == b_y.data)
            val_num += b_x.size(0)

        train_loss_all.append(train_loss/train_num)
        train_acc_all.append(train_correct/train_num)
        val_loss_all.append(val_loss/val_num)
        val_acc_all.append(val_correct/val_num)

        print("{} train loss:{:.4f} train acc:{:.4f}".format(epoch,train_loss_all[-1],train_acc_all[-1]))
        print("{} val loss:{:.4f} val acc:{:.4f}".format(epoch, val_loss_all[-1], val_acc_all[-1]))

        if val_acc_all[-1]>best_acc:
            best_acc=val_acc_all[-1]
            best_model_wts=copy.deepcopy(model.state_dict())

        time_use=time.time()-since
        print("time:",time_use)

        #保存
        model.load_state_dict(best_model_wts)
        torch.save(model.load_state_dict(best_model_wts),"/home/fthzzz/pycharm project1/LeNet_best.pth")

    train_process_show=pd.DataFrame(data={
        "epoch":range(num_epoch),
        "train_loss_all":train_loss_all,
        "val_loss_all":val_loss_all,
        "train_acc_all":train_acc_all,
        "val_acc_all":val_acc_all
    })
    return train_process_show

#展示数据
def MatPlotshow(train_process_show):
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(train_process_show["epoch"],train_process_show.train_loss_all,'ro-',label="train_loss")
    plt.plot(train_process_show["epoch"], train_process_show.val_loss_all, 'bs-', label="val_loss")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 2)
    plt.plot(train_process_show["epoch"], train_process_show.train_acc_all, 'ro-', label="train_acc")
    plt.plot(train_process_show["epoch"], train_process_show.val_acc_all, 'bs-', label="val_acc")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("acc")
    plt.legend()
    plt.show()


if __name__=="__main__":
    #实例化
    model=LeNet()
    train_dataloader,val_dataloader=train_val_process()
    train_process_show=model_train(model,train_dataloader,val_dataloader)
    MatPlotshow(train_process_show)










