import torch.nn as nn

# 定义模型结构
class SleepModel(nn.Module):

    def __init__(self, num_classes, is_training=True):
        super(SleepModel,self).__init__()
        self.is_training = is_training
        self.conv1 = nn.Conv1d(in_channels = 1,out_channels = 16,kernel_size = 3,stride = 1,padding = 1)
        self.conv2 = nn.Conv1d(16,32,3,1,1)
        self.conv3 = nn.Conv1d(32,64,3,1,1)
        self.conv4 = nn.Conv1d(64,64,5,1,2)
        self.conv5 = nn.Conv1d(64,128,5,1,2)
        self.conv6 = nn.Conv1d(128,128,5,1,2)
        self.maxpool = nn.MaxPool1d(3,stride=2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(0.5)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(40832,256)
        self.fc21 = nn.Linear(40832,16)
        self.fc22 = nn.Linear(16,256)
        self.fc3 = nn.Linear(256,num_classes)

    def forward(self,x):
        x = x.view(x.size(0),1,x.size(1))
        x = self.conv1(x)   #nn.Conv1d(in_channels = 1,out_channels = 32,kernel_size = 11,stride = 1,padding = 5)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv5(x)
        x = self.relu(x)
        x = self.conv6(x)
        x = self.relu(x)        
        x = self.maxpool(x)
        x = self.dropout(x)
        x = self.flatten(x)
        x1 = self.fc1(x)
        x1 = self.relu(x1)
        x21 = self.fc21(x)
        x22 = self.relu(x21)
        x22 = self.fc22(x22)
        x2 = self.relu(x22)
        x = self.fc3(x1+x2)
        return x