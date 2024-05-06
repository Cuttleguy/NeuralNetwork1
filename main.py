import torch
import torchvision
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch.optim as optim
from torchvision import transforms,datasets

train = datasets.MNIST("",train=True,download=True,
                       transform=transforms.Compose([transforms.ToTensor()]))

test = datasets.MNIST("",train=False,download=True,
                       transform=transforms.Compose([transforms.ToTensor()]))
trainset = torch.utils.data.DataLoader(train,batch_size=10,shuffle=True)
testset = torch.utils.data.DataLoader(test,batch_size=10,shuffle=True)
# for data in trainset:
#     print(data)
#     break
# x,y=data[0][0],data[1][0]
# print(y)
# print(x.shape)
# total=0
# counter_dict = {x:0 for x in range(10)}
# for data in trainset:
#     Xs, ys = data
#     for y in ys:
#         counter_dict[int(y)]+=1
#         total+=1
# print(counter_dict)
# for i in counter_dict:
#     print(f"{i}: {counter_dict[i]/total*100}%")
# plt.imshow(data[0][0].view(28,28))
# plt.show()

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 10)
    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return F.log_softmax(x,dim=1)



net=Net()
print(net)
X = torch.rand([28,28])
X = X.view([-1,28*28])
output=net(X)
print(output)
optimizer=optim.Adam(net.parameters(),lr=0.001)
