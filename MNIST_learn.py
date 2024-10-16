import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets,transforms
import numpy as np
import matplotlib.pyplot as plt
import time

epochs=10
steps=0
running_loss=0
losses=[]

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,),(0.5,))
])

trainset = datasets.MNIST(root='mnist_data',download=True,train=True,transform=transform)
trainloader = torch.utils.data.DataLoader(trainset,batch_size=64,shuffle=True)

testset = datasets.MNIST(root='mnist_data',download=True,train=False,transform=transform)
testloader = torch.utils.data.DataLoader(testset,batch_size=64,shuffle=False)

# def imshow(img):
#     img = img / 2 + 0.5
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg,(1,2,0)),cmap='gray')
#     plt.show()

# dataiter = iter(trainloader)
# images,labels = next(dataiter)

# imshow(torchvision.utils.make_grid(images[:4]))
# print('labels:',labels[:4].numpy())

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN,self).__init__()
        self.fc1=nn.Linear(28*28,128)
        self.relu=nn.ReLU()
        self.fc2=nn.Linear(128,10)
        self.log_softmax=nn.LogSoftmax(dim=1)

    def forward(self,x):
        x=x.view(x.shape[0],-1)
        x=self.fc1(x)
        x=self.relu(x)
        x=self.fc2(x)
        x=self.log_softmax(x)
        return x

model=SimpleNN()
print(model)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'模型参数数量：{count_parameters(model)}')

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(),lr=0.003)

for epoch in range(epochs):
    for images,labels in trainloader:
        steps+=1
        optimizer.zero_grad()
        log_ps=model(images)
        loss=criterion(log_ps,labels)
        loss.backward()
        optimizer.step()
        running_loss+=loss.item()
    else:
        epoch_loss=running_loss/len(trainloader)
        losses.append(epoch_loss)
        print(f'{epoch+1}/{epochs},{epoch_loss:.4f}')
        running_loss=0

plt.plot(range(1,epochs+1),losses,label='training loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('training loss over epoch')
plt.legend()
plt.show()