import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNPred(nn.Module):
    def __init__(self, features, days, markets):
        super(CNNPred, self).__init__()
        # If markets = 1, then this model becomes CNNPred2D else it becomes CNNPred3D
        self.conv1 = nn.Conv2d(in_channels = features, out_channels = 8, kernel_size = (1, 1), stride=1)
        self.conv2 = nn.Conv2d(in_channels = 8, out_channels = 8, kernel_size = (3, markets), stride=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size = (2, 1), stride=2)
        self.conv3 = nn.Conv2d(in_channels = 8, out_channels = 8, kernel_size = (3, 1), stride=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size = (2, 1), stride=2)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features = 8*int((int((days - 2)/2) - 2)/2), out_features = 2)
        self.featact = nn.ReLU()
        self.final = nn.Sigmoid()

    def forward(self, x):
        #Input should be (batch_size, features, days, markets)
        x = self.featact(self.conv1(x))
        x = self.featact(self.maxpool1(self.conv2(x)))
        x = self.featact(self.maxpool2(self.conv3(x)))
        x = self.flatten(x)
        x = self.final(self.fc(x))
        return x

class CNN_LSTM(nn.Module):
    def __init__(self, days, features, batch_size, hidden_size = 16):
        super(CNN_LSTM, self).__init__()
        self.batch_size = batch_size
        self.features = features
        self.hidden_size = hidden_size
        self.cnn = nn.Conv1d(in_channels = features, out_channels = 32, kernel_size = 1)
        self.lstm = nn.LSTM(input_size=32, hidden_size=self.hidden_size,
                                    num_layers=1, batch_first=True)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features = self.hidden_size*days, out_features = 1)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
    
    def forward(self, x):
        #Input should be (batch_size, features, days)
        x = self.relu(self.tanh(self.cnn(x)))
        x = torch.transpose(x, 1, 2)
        b = int(x.size()[0])
        ho = torch.zeros(1 ,b, self.hidden_size).cuda()
        co = torch.zeros(1 ,b, self.hidden_size).cuda()
        x, (_, _) = self.lstm(x, (co, ho))
        x = self.flatten(x)
        x = self.fc(x)
        return x
    
if __name__ == "__main__":
    a = CNNPred(32, 30, 5)
    b = CNN_LSTM(16 ,8, 64)

    u = torch.randn(64, 32, 30, 5)
    v = torch.randn(64, 8, 16)

    print(a(u).size())
    print(b(v).size())

        