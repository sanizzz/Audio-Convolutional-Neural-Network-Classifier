import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self,in_channels,out_channels,stride = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels,out_channels,3,stride,padding=1,bias=False) #bias false to save computing as batchnorm has it 
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels,out_channels,3,stride,padding=1,bias=False)
        self.bn2 = self.bn1 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        self.use_shortcut = stride != 1 or in_channels != out_channels
        if self.shortcut :
            self.shortcut = nn.Sequential(nn.Conv2d(in_channels,out_channels,1,stride=stride,bias=False),nn.BatchNorm2d(out_channels))
