mport torch
import torch.nn as nn
from ipdb import set_trace 


class my_Model(nn.Module):
    def __init__(self):
        super().__init__()

        #Start Block
        self.conv1= nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu= nn.ReLU(inplace=True)
        self.batchnorm1= nn.BatchNorm2d(num_features=64)
        self.conv2= nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.batchnorm2 = nn.BatchNorm2d(num_features=64)
        self.pooling1= nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.batchnorm3=nn.BatchNorm2d(num_features=64)
        
        
        
        #Block 1 convert image size to 24*24
        self.conv2a= nn.Conv2d(64,128,kernel_size=1, stride=1, padding=0)
        self.batchnorm4= nn.BatchNorm2d(num_features=128)
        self.conv2b= nn.Conv2d(128, 128, kernel_size=3,stride=1, padding=1)
        self.batchnorm5= nn.BatchNorm2d(num_features=128)
        self.pooling2a= nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.conv2c = nn.Conv2d(64,128, kernel_size=1, stride=1, padding=0)
        self.batchnorm6= nn.BatchNorm2d(num_features=128)
        self.conv2d = nn.Conv2d(128,128, kernel_size=3, stride=1, padding=1)
        self.batchnorm2d= nn.BatchNorm2d(num_features=128)
        #after cancatenation
        self.pooling2b= nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        #block 2 ci=overt image to 12*12
        self.conv3a= nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)
        self.batchnorm7= nn.BatchNorm2d(num_features=256)
        self.conv3b= nn.Conv2d(256,256, kernel_size=3, stride=1, padding=1)
        self.batchnorm8= nn.BatchNorm2d(num_features=256)
        self.pooling3a= nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.conv3c = nn.Conv2d(256,256, kernel_size=1, stride=1, padding=0)
        self.batchnorm9= nn.BatchNorm2d(num_features=256)
        self.conv3d = nn.Conv2d(256,256, kernel_size=3, stride=1, padding=1)
        self.batchnorm3d= nn.BatchNorm2d(num_features=256)
        #after can image become 6*6*512
        self.pooling3b= nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        #Block 3
        self.conv4a=nn.Conv2d(512,512, kernel_size=3,stride=1,padding=1)
        self.batchnorm10= nn.BatchNorm2d(num_features=512)
        #self.conv5a=nn.Conv2d(512,512,kernel_size=3,stride=1,padding=1)
        #self.batchnorm11=nn.BatchNorm2d(num_features=512)
        #self.conv6a=nn.Conv2d(512,512,kernel_size=3,stride=1,padding=1)
        #self.batchnorm12=nn.BatchNorm2d(num_features=512)
        self.pooling4a= nn.MaxPool2d(kernel_size=2, stride=2,padding=0)
        self.conv7a=nn.Conv2d(512,512,kernel_size=1,stride=1,padding=0)
        self.batchnorm13=nn.BatchNorm2d(num_features=512)
        #self.conv8a=nn.Conv2d(512,512,kernel_size=3,stride=1,padding=1)
        #self.batchnorm14=nn.BatchNorm2d(num_features=512)
        self.pooling5a= nn.MaxPool2d(kernel_size=3, stride=1,padding=1)
        self.poolingAvg= nn.AvgPool2d(kernel_size=2,stride=2,padding=0)
        self.fc1 =nn.Linear(in_features=512, out_features=512)
        self.dropout=nn.Dropout(0.5)
        self.fc2 =nn.Linear(in_features=512, out_features=256)
        self.dropout= nn.Dropout(0.7)
        self.fc3 = nn.Linear(in_features=256, out_features=7)

    def forward(self,x):
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.relu(x)
        x = self.pooling1(x)
        x = self.batchnorm3(x)
        #Route B
        x_2c = self.pooling2a(x)
        x_2c = self.conv2c(x_2c)
        x_2d = self.conv2b(x_2c)
        x_2d = self.batchnorm2d(x_2d)
        x_2d = self.conv2d(x_2c)
       # print("x_2c", x_2c.size())
        x_2c = self.batchnorm6(x_2d)
        x_2c = self.relu(x_2c)
        #block 1 route A
        x_2a = self.conv2a(x)
        x_2a = self.batchnorm4(x_2a)
        x_2a = self.relu(x_2a)
        x_2b = self.conv2b(x_2a)
        #print("x_2b", x_2b.size())
        x_2b = self.batchnorm5(x_2b)
        x_2b = self.relu(x_2b)
     
       
        #cancatenationee
        
        x = torch.cat([x_2b, x_2d], dim=1)
        x = self.pooling2b(x)
       # print("size of pooling2b", x.size())
        #Block 2 Route A
        x_3a = self.conv3a(x)
        x_3a = self.batchnorm7(x_3a)
        x_3a = self.relu(x_3a)
        x_3b = self.conv3b(x_3a)
        x_3b = self.batchnorm8(x_3b)
        x_3b = self.relu(x_3b)
       # print("size of conv3b", x_3b.size())
        #Route B
        x_3c = self.pooling3a(x)
       # print("size of pool 3a is", x_3c.size())
        x_3c = self.conv3c(x_3c)
       # print("size of conv 3c", x_3c.size())
        x_3c = self.batchnorm9(x_3c)
        x_3c = self.relu(x_3c)
        x_3d = self.conv3d(x_3c)
        x_3d = self.batchnorm3d(x_3d)
        x_3d = self.relu(x_3d)
        #Cancatenation
        x = torch.cat([x_3b, x_3d], dim=1)
       # print("size of cat2", x.size())
        x = self.pooling3b(x)
       # print("pool 3b", x.size())

        # block 4
        x_4a = self.conv4a(x)
        x_4a = self.batchnorm10(x_4a)
        x_4a = self.relu(x_4a)
        #x_5a = self.conv5a(x_4a)
        #x_5a = self.batchnorm11(x_5a)
        #x_5a = self.relu(x_5a)
        #x_6a = self.conv5a(x_5a)
        #x_6a = self.batchnorm12(x_6a)
       # x_6a = self.relu(x_6a)
        x_6a = self.pooling4a(x_4a)
        x_7a = self.conv7a(x_6a)
        x_7a = self.batchnorm13(x_7a)
        x_7a = self.relu(x_7a)
        #x_8a = self.conv8a(x_7a)
       # x_8a = self.batchnorm13(x_8a)
        #x_8a = self.relu(x_8a)
        x = self.pooling5a(x_7a)
        x = self.poolingAvg(x)
       # print("last pool",x.size())
        x = x.view(x.size(0), -1)
        
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x
