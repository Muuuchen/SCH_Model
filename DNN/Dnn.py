import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.autograd import Variable
import csv

file_path = '/home/muuuchen/Desktop/HRNet/SCH_Model/res/future.csv'
def get_data(file_path):
    with open(file_path, newline='') as file:
        reader = csv.reader(file)
        data_x = []
        data_y = []
        for list in reader:
            data_x.append([float(x) for x in list[0:-1]])
            data_y.append([1.0] if list[-1]=="True" else [0.0])
            #1 up  #     0 down
        return data_x,data_y

class Net(torch.nn.Module):
    def __init__(self,n_input,n_hidden,n_output):
        super(Net,self).__init__()
        self.hidden1 = torch.nn.Linear(n_input,n_hidden)
        self.relu1 = nn.ReLU()
        self.hidden2 = torch.nn.Linear(n_hidden,n_hidden)
        self.relu2 = nn.ReLU()
        self.predict = torch.nn.Linear(n_hidden,n_output)

    def forward(self, input):
        out = self.hidden1(input)
        out = self.relu1(out)
        out = self.hidden2(out)
        out = self.relu2(out)
        out = self.predict(out)
        return out
if __name__== "__main__":
    net = Net(4,20,1)

    optimizer = torch.optim.SGD(net.parameters(),lr=0.01)
    criterion = nn.BCEWithLogitsLoss()
    num_epoches = 4000
    x_data,y_data = get_data(file_path)
    x_data = torch.tensor(x_data)
    y_data = torch.tensor(y_data)


    for epoch in range(num_epoches):
        out = net(x_data)
        loss = criterion(out,y_data)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        out = torch.sigmoid(out)
        out_class = torch.round(out)
        accuracy = (out_class == y_data).sum().item() / y_data.size(0)
        if(epoch + 1) % 100 == 0:
            print('Epoch [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'.format(epoch + 1, num_epoches, loss.item(), accuracy * 100))
    torch.save(net.state_dict(), 'model.pth')
