import torch
import torch.nn as nn#所有的层都包括在nn里、包括所有的激活函数
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets,transforms,models
import torchsummary as summmary#展示模型参数

#卷积神经网络模块构建
class LeNet(nn.Module):#继承
    # 第一步初始化：需要网络层，需要激活函数
    def __init__(self):
        super(LeNet,self).__init__()
        #搭建第一层卷积/输入的通道数，输出的通道数，卷积核大小，边缘填充，步长
        self.c1=nn.Conv2d(in_channels=3,out_channels=6,kernel_size=5,padding=2,stride=1)
        #激活函数
        self.sig=nn.Sigmoid()
        #搭建池化层
        self.p1=nn.MaxPool2d(kernel_size=3,stride=1)#2d表示输入和输出的是二维数据 有疑问请看csdn收藏夹对3d的理解
        #从池化到最后的全连接层，有一个将所有特征值展开的操作
        self.flatten=nn.Flatten()
        #搭建全连接层/线性全连接层
        self.fc1=nn.Linear(400,120)#输入的特征数和输出的特征数
    #第二步：搭建网络前向传播过程
    def forward(self,x):#x表示输入
        x=self.sig(self.c1(x))#卷积和激活
        x=self.p1(x)
        x=self.flatten(x)
        x=self.fc1(x)
        return x


# if __name__=="__main__":
#     device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     #模型实例化/类对象
#     model_LeNet=LeNet().to(device)







