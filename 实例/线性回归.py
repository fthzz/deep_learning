import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
x=torch.rand(512)#均匀分布512个张量 [0,1) 相当于矩阵1*512
noise=0.2*torch.randn(x.size())#标准正态分布 高斯白噪音
w=3
b=10
y=w*x+b+noise
class linear(nn.Module):#父类
    def __init__(self,in_fea,out_fea):
        super(linear,self).__init__()#在子类中调用父类的初始化方法
        self.out=nn.Linear(in_features=in_fea,out_features=out_fea) #定义了一个神经网络的线性层
        #nn.Linear的功能是在定义了线性层后
        # 对tensor进行线性变换
        # 原理是：1.tensor的矩阵乘以权重的矩阵weights
               # 2.相乘之后的每个数减去误差bais
    def forward(self,x):#专门用来进行前向传播
        x=self.out(x)#输入tensor数据
        return x
input_x=torch.unsqueeze(x,dim=1)#上面radn完成的是 1个tensor tenor里有512个值 1*512
#实际上我们要完成的是512个tensor 每个tensor都有一个值 产生512*1的矩阵
#所以用torch.unsqueeze升一维
input_y=torch.unsqueeze(y,dim=1)
#实例化
model=linear(1,1)#in_fea的大小是你传入的每个tensor中的值的个数
                              #out_fea的大小是希望线性转化后输出的每个tensor所包含的值的个数

loss_func=nn.MSELoss()#损失函数 平均平方误差
#optimizer 优化器
#变量optimizer 能够保持当前参数状态并基于计算得到的梯度进行参数更新
optimizer=torch.optim.SGD(model.parameters(),lr=0.02)#torch.optim.SGD 随机梯度下降算法 运行时需要把它赋给一个变量
#model.parameters()储存的是权重和偏差/误差
plt.ion()#开启交互模式
for step in range(400):#每一次运行 都对512个tensor进行了处理
    # forward具有特殊性 在有多个方法时 也会被优先执行
    pred=model(input_x)#这句话等价 pred=model.forward(input_x)
                       # 传入之后pred 变为了预测的y值
    loss=loss_func(pred,input_y)#计算所得预测值y 与 真实的y 求均方差
    optimizer.zero_grad()#梯度初始化为0 是对上一次数据所产生的各参数的梯度清0
    loss.backward()#计算反向传播中各个参数的梯度
    optimizer.step()#通过梯度信息 对参数进行更新
    if step%20==0:
        plt.cla()#清除当前图像 重新绘制 主要用于动态更新
        plt.scatter(input_x.data.numpy(),input_y.data.numpy())#plt绘制的是采用numpy的数据
        plt.plot(input_x.data.numpy(),pred.data.numpy(),"r-",lw=4)#lw字体粗细
        [w,b]=model.parameters()#这里每组tensor刚好只有一个数据 所以权重和偏差都只有一个
        plt.xlim(0,1.1)
        plt.ylim(0,20)
        # .item()用于单个数值的tensor转为python数字 且精度更高
        plt.text(0,0.5,'loss=%.4f,k=%.2f,b=%.2f'%(loss.item(),w.item(),b.item()))
        plt.pause(1)#暂停一秒
plt.ioff()#恢复默认模式 默认模式下需要plt.show()才可以显示
plt.show()#如果不加这个 在循环结束后它会自动关闭




