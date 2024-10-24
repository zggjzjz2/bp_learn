import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets,transforms
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.metrics import confusion_matrix
import seaborn as sns

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
        x=x.view(x.size(0),-1)
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

def evaluate(model,testloader):
    total=0
    correct=0
    with torch.no_grad():
        for images,labels in testloader:
            log_ps=model(images)
            ps=torch.exp(log_ps)
            _,predicted=torch.max(ps,1)
            total+=labels.size(0)
            correct+=(predicted==labels).sum().item()
    accuracy=100*correct/total
    return accuracy

accuracy=evaluate(model,testloader)
print(f'accuracy:{accuracy:.2f}%')

def plot_confusion_matrix(model,testloader):
    all_pred=[]
    all_label=[]
    with torch.no_grad():
        for images,labels in testloader:
            log_ps=model(images)
            ps=torch.exp(log_ps)
            _,pred=torch.max(ps,1)
            all_pred.extend(pred.cpu().numpy())
            all_label.extend(labels.cpu().numpy())

    cm=confusion_matrix(all_label,all_pred)
    plt.figure(figsize=(10,8))
    sns.heatmap(cm,annot=True,fmt='d',cmap='Blues')
    plt.xlabel('predicted')
    plt.ylabel('true')
    plt.title('confusion_matrix')
    plt.show()

plot_confusion_matrix(model,testloader)