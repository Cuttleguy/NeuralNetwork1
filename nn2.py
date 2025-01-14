import os
import cv2
import numpy as np
from tqdm import tqdm
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
if(torch.cuda.is_available()):

    device = torch.device("cuda:0")
else:
    device= torch.device("cpu")
REBUILD_DATA = False

class DogsVSCats():
    IMG_SIZE = 50
    CATS = "PetImages/Cat"
    DOGS = "PetImages/Dog"
    LABELS = {CATS: 0, DOGS: 1}
    training_data = []
    catcount = 0
    dogcount = 0
    def make_training_data(self):
        for label in self.LABELS:

            print(label)
            for f in tqdm(os.listdir(label)):
                try:
                    path=os.path.join(label,f)
                    img=cv2.imread(path,cv2.IMREAD_GRAYSCALE)
                    img=cv2.resize(img,(self.IMG_SIZE,self.IMG_SIZE))
                    # print(np.array(img).shape)
                    # print(np.eye(2)[self.LABELS[label]].shape)
                    self.training_data.append([np.array(img), np.eye(2)[self.LABELS[label]]])
                    if label == self.CATS:
                        self.catcount+=1
                    elif label == self.DOGS:
                        self.dogcount+=1

                except Exception as e:
                    pass
        print(len(self.training_data))
        np.random.shuffle(self.training_data)
        print(len(self.training_data))
        # with open("training_data.pkl", "wb") as f:
        #     pickle.dump(self.training_data, f)
        np.save("training_data.npy",np.array(self.training_data,dtype=object))
        print("Cats: ", self.catcount)
        print("Dogs: ", self.dogcount)
if REBUILD_DATA:
    dogsvcats=DogsVSCats()
    dogsvcats.make_training_data()
training_data=np.load("training_data.npy",allow_pickle=True)
print(len(training_data))
class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1,32,5)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.conv3 = nn.Conv2d(64, 128, 5)

        x = torch.randn(50,50).view(-1,1,50,50)
        self._to_linear = None
        # self.convs(x)
        # print(self._to_linear)

        self.fc1= nn.Linear(512, 512)
        self.fc2 = nn.Linear(512,2)
    def convs(self,x):
        x = F.max_pool2d(F.relu(self.conv1(x)),(2,2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))
        # if(self._to_linear is None):
        #     self._to_linear=x[0].shape[0]*x[0].shape[1]*x[0].shape[2]
        return x
    def forward(self,x):
        x =self.convs(x)
        x = x.view(-1,512)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x,dim=1)


net=Net().to(device)
print(net)
optimizer= optim.Adam(net.parameters(),lr=0.001)
loss_function=nn.MSELoss()
X = torch.Tensor([i[0] for i in training_data]).view(-1,50,50).to(device)
X = X/255.0
y = torch.Tensor([i[1] for i in training_data]).to(device)
VAL_PCT= 0.1
val_size = int(len(X)*VAL_PCT)
print(val_size)
train_X = X[:-val_size]
train_y = y[:-val_size]
test_X = X[-val_size:]
test_y = y[-val_size:]
# print(len(train_X))
# print(len(test))

def train(net):
    BATCH_SIZE = 100
    EPOCHS = 17
    for epoch in range(EPOCHS):

        for i in tqdm(range(0, len(train_X), BATCH_SIZE)):
            # print(i,i+BATCH_SIZE)
            batch_X = train_X[i:i+BATCH_SIZE].view(-1,1,50,50).to(device)
            batch_y = train_y[i:i+BATCH_SIZE].to(device)
            optimizer.zero_grad()
            outputs= net(batch_X)
            loss=loss_function(outputs,batch_y)
            loss.backward()
            optimizer.step()
        print(epoch)
        print(loss)
def test(net):
    correct = 0
    total = 0
    with torch.no_grad():
        for i in tqdm(range(len(test_X))):
            real_class=torch.argmax(test_y[i]).to(device)
            net_out=net(test_X[i].view(-1,1,50,50))[0]
            predicted_class=torch.argmax(net_out)
            if predicted_class == real_class:
                correct+=1
            total+=1
    print("Accuracy: ", round(correct/total,3))
train(net)
test(net)