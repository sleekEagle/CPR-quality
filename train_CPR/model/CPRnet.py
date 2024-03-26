import torch
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.nn.functional as F

class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
class CPRnet(nn.Module):
    def __init__(self,conf):
        super(CPRnet, self).__init__()
        tr_conf=conf.model.transformer
        self.feature_extractor = AlexNet(num_classes=tr_conf.d_model)
       
        self.encoder_layers = TransformerEncoderLayer(tr_conf.d_model, tr_conf.nhead,
                                                      tr_conf.dim_feedforward,
                                                      tr_conf.dropout,
                                                      batch_first=True)
        self.transformer_encoder = TransformerEncoder(self.encoder_layers, tr_conf.n_layers)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        _,_,h,w=x.shape
        x=x.view(-1,1,h,w)
        feat=self.feature_extractor(x)
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = self.transformer_encoder(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
    
