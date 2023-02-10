import torch as t 
import torch.nn as nn
from collections import OrderedDict
from .basic_module import BasicModule 
import time 
import torchsnooper

#Priliminary reconstruction module
class recblock(BasicModule):
    def __init__(self,cr,size):
        super(recblock,self).__init__()

        self.cr = cr        #1/16 = M/N
        self.size = size    #block_size -> n = 16
                            #N = n*n
        #input: (361, 16)
        self.fc1 = nn.Linear(int(self.cr*self.size*self.size),self.size*self.size) #(in_channel, out_channel) = (M, N) = (16, 256)
        #output: (361, 256)
        cnn_layers = OrderedDict()
        cnn_layers['conv1'] = nn.Conv2d(1,128,1,1,0)
        cnn_layers['relu1'] = nn.ReLU(inplace=True)
        cnn_layers['conv2'] = nn.Conv2d(128,64,1,1,0)
        cnn_layers['relu2'] = nn.ReLU(inplace=True)
        cnn_layers['conv3'] = nn.Conv2d(64,32,3,1,1)
        cnn_layers['relu3'] = nn.ReLU(inplace=True)
        cnn_layers['conv4'] = nn.Conv2d(32,16,3,1,1)
        cnn_layers['relu4'] = nn.ReLU(inplace=True)
        cnn_layers['conv5'] = nn.Conv2d(16,1,3,1,1)
        self.rec_cnn = nn.Sequential(cnn_layers)

    def forward(self,input):
        #inputはmeasurement Yの事
        b_s = input.size(0) 
        #print(input.shape) # (361, 16)
        #print(f"b_s -> {b_s}") #361
        x_1 = self.fc1(input).view(b_s,self.size,self.size).unsqueeze_(1)
        #print(x_1.shape)
        output = self.rec_cnn(x_1).view(b_s,self.size,self.size)
        return output

#↓ここから   
#Residual   
class resblock(BasicModule):
    def __init__(self,cr,size):
        super(resblock,self).__init__()

        self.cr = cr
        self.size = size
        #print(f"res_block -> {self.cr*self.size*self.size}") # 16.0
        self.fc1 = nn.Linear(int(self.cr*self.size*self.size),self.size*self.size)
        cnn_layers = OrderedDict()
        cnn_layers['conv1'] = nn.Conv2d(1,64,1,1,0)
        cnn_layers['relu1'] = nn.ReLU(inplace=True)
        cnn_layers['conv2'] = nn.Conv2d(64,32,3,1,1)
        cnn_layers['relu2'] = nn.ReLU(inplace=True)
        cnn_layers['conv3'] = nn.Conv2d(32,1,3,1,1)
        self.rec_cnn = nn.Sequential(cnn_layers)

    def forward(self,input):
        b_s = input.size(0)
        x_1 = self.fc1(input).view(b_s,self.size,self.size).unsqueeze_(1)
        output = self.rec_cnn(x_1).view(b_s,self.size,self.size)
        return output

#Motion compensation
class MCblock(BasicModule):
    def __init__(self,m_matrix,cr,blk_size,ref_size):
        super(MCblock,self).__init__()

        #print(f"cr -> {cr}, m_matrix -> {m_matrix.shape}, blk_size -> {blk_size}, ref_size -> {ref_size}")
        #cr -> 0.0625, m_matrix -> (16, 256) = (M, N), blk_size -> 16, ref_size -> 32
        #m_matrix = measurement matrix
        
        self.cr = cr
        self.m = m_matrix
        self.blk_size = blk_size
        self.ref_size = ref_size
        
        self.rec = recblock(self.cr,self.blk_size)
        self.res = resblock(self.cr,self.blk_size)
        self.fc = nn.Linear(self.ref_size*self.ref_size,self.blk_size*self.blk_size)

    def forward(self,input,ref,y):
        #print(f"input.size(0) -> {input.size(0)}")
        #print(f"ref_size -> {ref.shape}") #(361, 32, 32)
        b_s = input.size(0) # 361
        #input (361, 16)
        x_1 = self.rec(input)
        #x_1 (361, 16, 16)
        x_mc = self.fc(ref.view(b_s,self.ref_size*self.ref_size)).view(b_s,self.blk_size,self.blk_size)
        #x_mc (361, 16, 16)
        x_2 = x_mc.view(b_s,self.blk_size*self.blk_size).unsqueeze_(2)
        #x_2 (361, 256, 1)
        weight = self.m.repeat(b_s,1,1)
        #weight (361, 16, 256)
        y_mc = t.bmm(weight,x_2).squeeze_(2)
        #y_mc (361, 16)
        y_r = y_mc-y
        #y_r (361, 16) -> y residual
        output = self.res(y_r)+x_1
        
        return output,x_mc

#ここから    
class MCNet(BasicModule):
    def __init__(self,m_matrix,cr,blk_size,ref_size):
        super(MCNet,self).__init__()

        self.m = m_matrix #(16, 256)
        self.cr = cr #M/N
        self.blk_size = blk_size #n=16
        self.ref_size = ref_size #32

        self.block1 = MCblock(self.m,self.cr,self.blk_size,self.ref_size)
        self.block2 = MCblock(self.m,self.cr,self.blk_size,self.ref_size)
        self.block3 = MCblock(self.m,self.cr,self.blk_size,self.ref_size)
        self.block4 = MCblock(self.m,self.cr,self.blk_size,self.ref_size)

    def forward(self,input,ref,y):
        b_s = input.size(0)
        x_1,x_mc_1 = self.block1(input,ref,y)
        #x_1 (361, 16, 16)
        weight = self.m.repeat(b_s,1,1)
        input_2 = t.bmm(weight,x_1.view(b_s,self.blk_size*self.blk_size,1)).squeeze_(2)
        #input_2 (361, 16)
        x_2,x_mc_2 = self.block2(input_2,ref,y)
        #x_2 (361, 16, 16)
        input_3 = t.bmm(weight,x_2.view(b_s,self.blk_size*self.blk_size,1)).squeeze_(2)
        x_3,x_mc_3 = self.block3(input_3,ref,y)
        #x_3 (361, 16, 16)
        input_4 = t.bmm(weight,x_3.view(b_s,self.blk_size*self.blk_size,1)).squeeze_(2)
        output,x_mc_4 = self.block4(input_4,ref,y)
        #output (361, 16, 16)
        output_mc = (x_mc_1 + x_mc_2 + x_mc_3 + x_mc_4)/4.0
        return output,output_mc

class IPM(BasicModule):
    def __init__(self, num_channel=1):
        super().__init__()
        self.conv1 = nn.Conv2d(num_channel, 545, kernel_size=32, stride=32, padding=0, bias=False)
        
    def forward(self, x):
        x = self.conv1(x)
        
        return x


