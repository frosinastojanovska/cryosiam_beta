import torch.nn as nn


class CNNHead(nn.Module):
    """
    Build a CnnHead model.
    """

    def __init__(self, n_input_channels=1, n_output_channels=1, spatial_dims=3,
                 filters=(32, 64), kernel_size=3, padding=1):
        super().__init__()

        if spatial_dims == 2:
            conv = nn.Conv2d
            norm = nn.BatchNorm2d
        else:
            conv = nn.Conv3d
            norm = nn.BatchNorm3d
        self.cnn_head = nn.Sequential(conv(n_input_channels, filters[0], kernel_size, padding=padding),
                                      norm(filters[0]),
                                      nn.ReLU(inplace=False),
                                      conv(filters[0], filters[1], kernel_size, padding=padding),
                                      norm(filters[1]),
                                      nn.ReLU(inplace=False),
                                      conv(filters[1], n_output_channels, kernel_size, padding=padding))

    def forward(self, x):
        output = self.cnn_head(x)
        return output
