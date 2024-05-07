import torch
from torch import nn

def nms_1d(signal, window_size=3, threshold=0):
    """
    Perform Non-Maximum Suppression on a 1D signal.

    Parameters:
    - signal: A 1D PyTorch tensor containing the signal values.
    - window_size: The size of the neighborhood to compare for non-maximum suppression.
    - threshold: Minimum value to be considered a peak. Values below this are immediately suppressed.

    Returns:
    - A tensor of the same shape as `signal`, where non-maximum and low-threshold values are suppressed.
    """
    # Ensure window_size is odd to have a symmetric window
    assert window_size % 2 == 1, "Window size must be odd"
    
    half_window = window_size // 2
    padded_signal = torch.nn.functional.pad(signal, (half_window, half_window), mode='constant', value=0)
    peaks = torch.zeros_like(signal)
    
    for i in range(len(signal)):
        window = padded_signal[:,i:i + window_size]
        max_value = window.max()
        if signal[i] >= max_value and signal[i] > threshold:
            peaks[i] = signal[i]
    
    return peaks

# Example usage
# signal = torch.tensor([0.2, 0.9, 0.5, 0.4, 1.0, 0.7, 0.2, 0.3], dtype=torch.float32)
# window_size = 3  # Consider the element and one neighbor on each side
# threshold = 0.5  # Minimum peak value
# peaks = nms_1d(signal, window_size, threshold)

class SWNET(nn.Module):
    def __init__(self,conf):
        super(SWNET, self).__init__()
        #********head 1*******
        self.l1_1 = nn.Sequential(
            nn.Conv1d(in_channels=9, out_channels=32, kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=1)
        )
        self.l1_2 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=1)
        )
        
        self.lstm1 = nn.LSTM(input_size=64, hidden_size=64, num_layers=1, batch_first=True)
        self.fc_signal = nn.Linear(64, 1)
        self.fc_depth = nn.Linear(64, 1)
        self.sm=nn.Softmax(dim=1)

        self.peak_detection = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
        )

        # #**********head 2**********
        # self.l2_1 = nn.Sequential(
        #     nn.Conv1d(in_channels=9, out_channels=32, kernel_size=5, stride=2, padding=0),
        #     nn.ReLU(),
        #     nn.MaxPool1d(kernel_size=4, stride=3)
        # )
        # self.l2_2 = nn.Sequential(
        #     nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=0),
        #     nn.ReLU(),
        #     nn.MaxPool1d(kernel_size=4, stride=3)
        # )

        # #**********head 3**********
        # self.l3_1 = nn.Sequential(
        #     nn.Conv1d(in_channels=9, out_channels=32, kernel_size=7, stride=2, padding=0),
        #     nn.ReLU(),
        #     nn.MaxPool1d(kernel_size=4, stride=3)
        # )
        # self.l3_2 = nn.Sequential(
        #     nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=0),
        #     nn.ReLU(),
        #     nn.MaxPool1d(kernel_size=4, stride=3)
        # )

        # #**********classifier**********
        # if conf.smartwatch.seq_model=='CNN':
        #     self.drop_out = nn.Dropout(p=0.2)

        #     self.fc_n = nn.Linear(64 * 3, 1)
        #     self.fc_depth = nn.Linear(64 * 3, 1)
        #     self.sigmoid = nn.Sigmoid() 
        # elif conf.smartwatch.seq_model=='LSTM':
        #     self.lstm1 = nn.LSTM(input_size=64, hidden_size=64, num_layers=1, batch_first=True)
        #     self.lstm2 = nn.LSTM(input_size=64, hidden_size=64, num_layers=1, batch_first=True)
        #     self.lstm3 = nn.LSTM(input_size=64, hidden_size=64, num_layers=1, batch_first=True)
        #     self.fc_n = nn.Linear(64, 1)
        #     self.fc_depth = nn.Linear(64, 1)
        #     self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out1 = self.l1_1(x)
        out1 = self.l1_2(out1)
        out1 = out1.permute(0, 2, 1)

        lstm_out, (hn, cn) = self.lstm1(out1)
        bs,seq_len,hidden_dim=lstm_out.shape
        output=lstm_out.reshape(-1,hidden_dim)

        #reconstruct the original signal
        signal=self.fc_signal(output)
        signal=signal.reshape(bs,seq_len,1)
        signal=signal.swapaxes(1,2)
        signal_interpolated = torch.nn.functional.interpolate(signal, size=(300,), mode='linear').squeeze()

        #predict depth
        pred_depth=self.fc_depth(lstm_out[:,-1,:])
        
  # n_peaks=nms_1d(peaks_interpolated,window_size=9,threshold=0.7)
        # peak_out=self.peak_detection(peaks)
        # valley_out=self.peak_detection(valleys)
        # out1 = output[:, -1, :]
        # pred_n=self.fc_n(out1)
        # pred_depth=self.fc_depth(out1)

        # out3 = self.l3_1(x)
        # out3 = self.l3_2(out3)

        # out = torch.cat((out1, out2, out3), dim=1)

        # #classification
        # out = self.drop_out(out)
        # pred_n = self.sigmoid(self.fc_n(out))
        # pred_depth = self.fc_depth(out)


        return signal_interpolated,pred_depth
