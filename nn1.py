import torch
import torchvision
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch.optim as optim
from torchvision import transforms,datasets
import os
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
if(os.path.exists("models/nn1.pth")):
    lstate_dict=torch.load("models/nn1.pth")
    net.load_state_dict(lstate_dict)

print(net)
# X = torch.rand([28,28])
# X = X.view([-1,28*28])
# output=net(X)
# print(output)
optimizer=optim.Adam(net.parameters(),lr=0.001)

EPOCHS = 1
# for epoch in range(EPOCHS):
#     for data in trainset:
#         # data is a batch of featuresets and labels
#         X, y = data
#         net.zero_grad()
#         output=net(X.view(-1,28*28))
#         loss=F.nll_loss(output,y)
#         loss.backward()
#         optimizer.step()

correct = 0
total = 0
with torch.no_grad():
    for data in testset:
        X,y=data
        output=net(X.view(-1,784))
        for idx, i in enumerate(output):
            if torch.argmax(i) == y[idx]:
                correct+=1
            else:
                print("Predicted:")
                print(torch.argmax(i))
                print("Actual:")
                print(y[idx])
            total += 1

print("Accuracy: ",round(correct/total,3))
plt.imshow(X[0].view(28,28))
print("Expected Value: "+str(y[0].item()))
print(torch.argmax(net(X[0].view(-1,784))[0]))
plt.show()
# torch.save(net.state_dict(),"models/nn1.pth")
