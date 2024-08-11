import torch.nn as nn
import torch

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, pad=1, stride=1, bias=True):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=bias),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return self.block(x)

class BallTrackerNet(nn.Module):
    def __init__(self, out_channels=256):
        super().__init__()
        self.out_channels = out_channels

        self.conv1 = ConvBlock(in_channels=9, out_channels=64)
        self.conv2 = ConvBlock(in_channels=64, out_channels=64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = ConvBlock(in_channels=64, out_channels=128)
        self.conv4 = ConvBlock(in_channels=128, out_channels=128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv5 = ConvBlock(in_channels=128, out_channels=256)
        self.conv6 = ConvBlock(in_channels=256, out_channels=256)
        self.conv7 = ConvBlock(in_channels=256, out_channels=256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv8 = ConvBlock(in_channels=256, out_channels=512)
        self.conv9 = ConvBlock(in_channels=512, out_channels=512)
        self.conv10 = ConvBlock(in_channels=512, out_channels=512)
        self.ups1 = nn.Upsample(scale_factor=2)
        self.conv11 = ConvBlock(in_channels=512, out_channels=256)
        self.conv12 = ConvBlock(in_channels=256, out_channels=256)
        self.conv13 = ConvBlock(in_channels=256, out_channels=256)
        self.ups2 = nn.Upsample(scale_factor=2)
        self.conv14 = ConvBlock(in_channels=256, out_channels=128)
        self.conv15 = ConvBlock(in_channels=128, out_channels=128)
        self.ups3 = nn.Upsample(scale_factor=2)
        self.conv16 = ConvBlock(in_channels=128, out_channels=64)
        self.conv17 = ConvBlock(in_channels=64, out_channels=64)
        self.conv18 = ConvBlock(in_channels=64, out_channels=self.out_channels)

        self.softmax = nn.Softmax(dim=1)
        self._init_weights()
                  
    def forward(self, x, testing=False): 
        batch_size = x.size(0)
        print('batch size', batch_size)
        print('input size', x.shape)
        x = self.conv1(x)
        print('conv1', x.shape)
        x = self.conv2(x)
        print('conv2', x.shape)
        x = self.pool1(x)
        print('pool1', x.shape)
        x = self.conv3(x)
        print('conv3', x.shape)
        x = self.conv4(x)
        print('conv4', x.shape)
        x = self.pool2(x)
        print('pool2', x.shape)
        x = self.conv5(x)
        print('conv5', x.shape)
        x = self.conv6(x)
        print('conv6', x.shape)
        x = self.conv7(x)
        print('conv7', x.shape)
        x = self.pool3(x)
        print('pool3', x.shape)
        x = self.conv8(x)
        print('conv8', x.shape)
        x = self.conv9(x)
        print('conv9', x.shape)
        x = self.conv10(x)
        print('conv10', x.shape)
        x = self.ups1(x)
        print('ups1', x.shape)
        x = self.conv11(x)
        print('conv11', x.shape)
        x = self.conv12(x)
        print('conv12', x.shape)
        x = self.conv13(x)
        print('conv13', x.shape)
        x = self.ups2(x)
        print('ups2', x.shape)
        x = self.conv14(x)
        print('conv14', x.shape)
        x = self.conv15(x)
        print('conv15', x.shape)
        x = self.ups3(x)
        print('ups3', x.shape)
        x = self.conv16(x)
        print('conv16', x.shape)
        x = self.conv17(x)
        print('conv17', x.shape)
        x = self.conv18(x)
        print('conv18', x.shape)
        # x = self.softmax(x)
        out = x.reshape(batch_size, self.out_channels, -1)
        print('out', out.shape)
        if testing:
            out = self.softmax(out)
        return out                       
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.uniform_(module.weight, -0.05, 0.05)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)    
    
    
if __name__ == '__main__':
    device = 'cpu'
    model = BallTrackerNet().to(device)
    inp = torch.rand(1, 9, 360, 640)
    out = model(inp)
    print('out = {}'.format(out.shape))
    
    
