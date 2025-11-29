import torch.nn as nn
import torch.nn.functional as F
import torch

class VAE(nn.Module):

    def __init__(self, observation_shape=()):
        super().__init__()

        # print(observation_shape[-1])
        conv_output_dim = 64

        self.conv1 = nn.Conv2d(observation_shape[-1], 32, kernel_size=8, stride=4, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=2, stride=1, padding=1)

        self.compression = nn.Linear(conv_output_dim, conv_output_dim)

        self.flatten = torch.nn.Flatten()

        self.deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=1, padding=1)
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=1, padding=1)
        self.deconv3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.deconv4 = nn.ConvTranspose2d(32, observation_shape[-1], kernel_size=8, stride=4, padding=2)
        # self.conv3 = nn.Conv2d()

        print("VAE network initialized.")

    def _conv_forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        
        return x


    def _deconv_forward(self, x):
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        x = F.relu(self.deconv4(x))
        
        return x
        
    
    def forward(self, x):
        x = x / 255.0

        # print(x.mean())

        x = self._conv_forward(x)
    
        x = self.flatten(x)

        print(x)
        # x = self.compression(x)

        # Saving off the compression layer as an output
        enc = x

        # x = self._deconv_forward(x)

        return x, enc 








        

