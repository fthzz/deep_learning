import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
#linspace 线性等分
x=torch.unsqueeze(torch.linspace(-2,2,256),1)#均匀分布n个点
noise=0.2*torch.randn(x.size())
y=x.pow(2)+noise
class MLP(nn.Module):
    def __init__(self,in_fea,n_hidden,out_fea):
        super(MLP,self).__init__()
        self.hidden=nn.Linear(in_features=in_fea,out_features=n_hidden)
        self.relu=nn.ReLU()#激活函数 将数据小于0的部分置为0 其他不变
        self.out=nn.Linear(in_features=n_hidden,out_features=out_fea)

    def forward(self,x):
        x=self.relu(self.hidden(x))
        x=self.out(x)
        return x
model=MLP(1,100,1)

loss_func=nn.MSELoss()
optimizer=torch.optim.SGD(model.parameters(),lr=0.1)

plt.ion()
for step in range(400):
    pred=model(x)
    loss=loss_func(pred,y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if step%20==0:
        plt.cla()
        plt.scatter(x.data.numpy(),y.data.numpy())
        plt.plot(x.data.numpy(),pred.data.numpy(),'r-',lw=4)
        plt.text(0,0.5,'loss=%.4f'%(loss.item()))
        plt.pause(1)
plt.ioff()
plt.show()




