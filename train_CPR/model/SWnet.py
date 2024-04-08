import torch
from torch import nn

class SWNET(nn.Module):
    def __init__(self,conf):
        super(SWNET, self).__init__()
        #********head 1*******
        self.l1_1 = nn.Sequential(
            nn.Conv1d(in_channels=9, out_channels=32, kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=3)
        )
        self.l1_2 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=3)
        )

        #**********head 2**********
        self.l2_1 = nn.Sequential(
            nn.Conv1d(in_channels=9, out_channels=32, kernel_size=5, stride=2, padding=0),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=3)
        )
        self.l2_2 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=3)
        )

        #**********head 3**********
        self.l3_1 = nn.Sequential(
            nn.Conv1d(in_channels=9, out_channels=32, kernel_size=7, stride=2, padding=0),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=3)
        )
        self.l3_2 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=3)
        )

        #**********classifier**********
        self.drop_out = nn.Dropout(p=0.2)

        self.fc1 = nn.Linear(64 * 3, 1000)
        self.fc2 = nn.Linear(64 * 3, 1000)


        self.layer2 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.drop_out = nn.Dropout()
        self.fc_n = nn.Linear(64 * 3, 1)
        self.fc_depth = nn.Linear(64 * 3, 1)

    def forward(self, x):
        out1 = self.l1_1(x)
        out1 = self.l1_2(out1).mean(dim=2)

        out2 = self.l2_1(x)
        out2 = self.l2_2(out2).mean(dim=2)

        out3 = self.l3_1(x)
        out3 = self.l3_2(out3).mean(dim=2)

        print('out1:', out1.shape)
        print('out2:', out2.shape)
        print('out3:', out3.shape)

        out = torch.cat((out1, out2, out3), dim=1)

        #classification
        out = self.drop_out(out)
        pred_n = self.fc_n(out)
        pred_depth = self.fc_depth(out)
        return pred_n.squeeze(), pred_depth.squeeze()
