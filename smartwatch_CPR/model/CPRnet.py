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
    
class FeatureNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(FeatureNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv1d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2),
            nn.Conv1d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2),
            nn.Conv1d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 1 * 1, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x=x.flatten(1)
        x = self.classifier(x)
        return x
    
class CPRnet(nn.Module):
    def __init__(self,conf):
        super(CPRnet, self).__init__()
        tr_conf=conf.model.transformer
        if conf.type=='image':
            self.feature_extractor = AlexNet(num_classes=tr_conf.d_model).double()
        elif conf.type=='XYZ':
            self.feature_extractor = FeatureNet(num_classes=tr_conf.d_model).double()
       
        self.encoder_layers = TransformerEncoderLayer(tr_conf.d_model, tr_conf.nhead,
                                                      tr_conf.dim_feedforward,
                                                      tr_conf.dropout,
                                                      batch_first=True)
        self.transformer_encoder = TransformerEncoder(self.encoder_layers, tr_conf.n_layers).double()
        self.lin = nn.Linear(tr_conf.dim_feedforward, 1).double()

    def forward(self, x):
        _,_,h,w=x.shape
        x=x.view(-1,h,w)
        feat=self.feature_extractor(x)
        enc_feat=self.transformer_encoder(feat)
        depth_pred=self.lin(enc_feat).squeeze()
        return depth_pred
    
