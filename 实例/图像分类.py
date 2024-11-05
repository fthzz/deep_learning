import cv2 as cv
import torch.optim as optim
import torchvision
from torchvision import transforms,models,datasets
import torch
import numpy as np
import matplotlib as plt
import os
from torch import nn
import imageio
import time
import warnings
import random
import sys
import copy
import json
from PIL import Image
import torch.utils.data

#GPU
train_GPU=torch.cuda.is_available()
if train_GPU:
    print("gpu")
else:
    print("error")
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


#导入数据
data_dir='/home/fthzzz/Desktop/ecar-vision/data'
train_dir='/home/fthzzz/Desktop/ecar-vision/data/train'
val_dir='/home/fthzzz/Desktop/ecar-vision/data/valid'

#图像增广和预处理
data_transforms={
    'train':
        transforms.Compose([
        transforms.RandomRotation(45),
        transforms.CenterCrop(304),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2,contrast=0.1,saturation=0.1,hue=0.1),
        transforms.RandomGrayscale(p=0.025),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ]),
    'valid':
        transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
}

#从外部文件夹读入
#ImageFolder是一个数据加载器
image_datasets={x:datasets.ImageFolder(os.path.join(data_dir,x),trainsform=data_transforms[x]) for x in ['train','valid']}
#载入数据
dataloaders={x:torch.utils.data.DataLoader(image_datasets[x],batch_size=8,shuffle=True) for x in ['train','valid']}


#初始化变量
model_name='resnet'
# feature_extract=True
filename='best.pth'


#模型实例化
model_ft=models.resnet50()

#冻结参数
#除了全连接输出层out_features（即输出的实际类别，这个是看自己的数据集有几个labels） 其它层的权重参数都一直保持不变
def set_parameter_requires_grad(model,feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad=False#反向传播是否计算梯度

#将输出层改为自己的输出层
def initialize_model(model_name,num_classes,feature_extract,use_pretrained=True):
    model_ft=models.resnet50(pretrained=use_pretrained)#pretrained参数=True 表示会下载模型提前训练好的参数 这些参数在模型中表现是比较好的
                                                       #=False 则是使用随机权重
    #调用函数
    set_parameter_requires_grad(model_ft,feature_extract)
    #读取原本Linear中的输入特征
    num_ftrs=model_ft.fc.in_features
    #更新自己的num_class
    model_ft.fc=nn.Linear(num_ftrs,num_classes)

    return model_ft


feature_extract=True

model_ft=initialize_model(model_name,5,feature_extract,use_pretrained=True)
model_ft=model_ft.to(device)

#保存所有的参数
params_to_update=model_ft.parameters()
#因为我们改了最后一层输出层，所以对于输出层的参数，我们是要更新的
if feature_extract:
    params_to_update=[]
    for name,param in model_ft.named_parameters():
        #我们将所有的require_gard设置为了False 唯独最后一个输出层
        #输出层的默认require_gard是True，将它存到parameters中
        if param.requires_grad==True:
            params_to_update.append(param)

#优化器
#优化器中放的是权重和偏置 一般是model.parameters
#优化器中的参数是需要更新的
optimizer_ft=optim.Adam(params_to_update,lr=0.01)
#学习率调整函数
scheduler=optim.lr_scheduler.StepLR(optimizer_ft,step_size=10,gamma=0.1)
criterion=nn.CrossEntropyLoss()


def train_model(model,dataloaders,criterion,optimizer,num_epoch=20):
    model.to(device)
    best_acc=0.0
    best_model_wts=copy.deepcopy(model.state_dict())

    for epoch in range(num_epoch):
        print(epoch+1, '/', num_epoch)
        for phase in ["train","valid"]:
            if phase=="train":
                model.train()
            else:
                model.eval()

            running_loss=0.0
            running_corrects=0

            for inputs,labels in dataloaders[phase]:
                inputs=inputs.to(device)
                labels=labels.to(device)

                outputs=model(inputs)
                loss=criterion(outputs,labels)
                _,preds=torch.max(outputs,1)
                optimizer.zero_grad()

                if phase=="train":
                    loss.backward()
                    optimizer.step()
                running_loss+=loss.item()*inputs.size(0)
                running_corrects+=torch.sum(preds==labels.data)

            #计算本轮损失
            epoch_loss=running_loss/len(dataloaders[phase].dataset)
            #计算本轮准确率
            epoch_acc=float(running_corrects)/len(dataloaders[phase].dataset)
            if phase=="train":
                print("train_epoch_acc=",epoch_acc)
            else:
                print("val_epoch_acc=",epoch_acc)

            #更新准确率
            if phase=="valid" and epoch_acc>best_acc:
                best_acc=epoch_acc
                best_model_wts=copy.deepcopy(model.state_dict())

                #保存参数
                state={
                    "state_dict":model.state_dict(),
                    "best_acc":best_acc,
                    "optimizer":optimizer.state_dict(),
                }
                torch.save(state,"/home/fthzzz/Desktop/ecar-vision/best.pt")#保存参数
        scheduler.step()
        print('-'*10)

    print("best_acc=",best_acc)
    model.load_state_dict(best_model_wts)
    #torch.save(model,'/home/fthzzz/Desktop/ecar-vision/best.pt')
    return model

model_ft=train_model(model_ft,dataloaders,criterion,optimizer_ft)



#再次训练
# print("train again")
# for param in model_ft.parameters():
#     param.requires_grad=True
# optimizer=optim.Adam(model_ft.parameters(),lr=0.01)
# scheduler=optim.lr_scheduler.StepLR(optimizer_ft,step_size=7,gamma=0.1)
# criterion=nn.CrossEntropyLoss()

#导入模型
#方法一：
#model_ft=torch.load('best.pt')#重载整个模型
#方法二：
# checkpoint=torch.load('best.pt')
# best_acc=checkpoint['best_acc']
# model_ft.load_state_dict(checkpoint['state_dict'])

# model_ft=train_model(model_ft,dataloaders,criterion,optimizer_ft)





















