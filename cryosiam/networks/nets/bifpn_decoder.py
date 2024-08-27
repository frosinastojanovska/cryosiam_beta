import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseConvBlock(nn.Module):
    """
    Depthwise separable convolution.
    """
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, spatial_dims=2):
        super(DepthwiseConvBlock, self).__init__()
        if spatial_dims == 3:
            conv = nn.Conv3d
            norm = nn.BatchNorm3d
        else:
            conv = nn.Conv2d
            norm = nn.BatchNorm2d

        self.depthwise = conv(in_channels, in_channels, kernel_size, stride,
                              padding, dilation, groups=in_channels, bias=False)
        self.pointwise = conv(in_channels, out_channels, kernel_size=1,
                              stride=1, padding=0, dilation=1, groups=1, bias=False)

        self.bn = norm(out_channels, momentum=0.9997, eps=4e-5)
        self.act = nn.ReLU()

    def forward(self, inputs):
        x = self.depthwise(inputs)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.act(x)


class ConvBlock(nn.Module):
    """
    Convolution block with Batch Normalization and ReLU activation.
    """

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, spatial_dims=2):
        super(ConvBlock, self).__init__()

        if spatial_dims == 3:
            conv = nn.Conv3d
            norm = nn.BatchNorm3d
        else:
            conv = nn.Conv2d
            norm = nn.BatchNorm2d

        self.conv = conv(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bn = norm(out_channels, momentum=0.9997, eps=4e-5)
        self.act = nn.ReLU()

    def forward(self, inputs):
        x = self.conv(inputs)
        x = self.bn(x)
        return self.act(x)


class BiFPNBlock(nn.Module):
    """
    Bi-directional Feature Pyramid Network
    """

    def __init__(self, feature_size=64, epsilon=0.0001, spatial_dims=2):
        super(BiFPNBlock, self).__init__()
        self.epsilon = epsilon

        self.p2_td = DepthwiseConvBlock(feature_size, feature_size, spatial_dims=spatial_dims)
        self.p3_td = DepthwiseConvBlock(feature_size, feature_size, spatial_dims=spatial_dims)
        self.p4_td = DepthwiseConvBlock(feature_size, feature_size, spatial_dims=spatial_dims)
        self.p5_td = DepthwiseConvBlock(feature_size, feature_size, spatial_dims=spatial_dims)

        self.p3_out = DepthwiseConvBlock(feature_size, feature_size, spatial_dims=spatial_dims)
        self.p4_out = DepthwiseConvBlock(feature_size, feature_size, spatial_dims=spatial_dims)
        self.p5_out = DepthwiseConvBlock(feature_size, feature_size, spatial_dims=spatial_dims)
        self.p6_out = DepthwiseConvBlock(feature_size, feature_size, spatial_dims=spatial_dims)

        self.w1 = nn.Parameter(torch.ones([2, 4], requires_grad=True))
        self.w1_relu = nn.ReLU()
        self.w2 = nn.Parameter(torch.ones([3, 4], requires_grad=True))
        self.w2_relu = nn.ReLU()

    def forward(self, inputs):
        p2_x, p3_x, p4_x, p5_x, p6_x = inputs

        # Calculate Top-Down Pathway
        w1 = self.w1_relu(self.w1)
        w1 = w1.clone() / (torch.sum(w1, dim=0) + self.epsilon)
        w2 = self.w2_relu(self.w2)
        w2 = w2.clone() / (torch.sum(w2, dim=0) + self.epsilon)

        p6_td = p6_x
        p5_td = self.p5_td(w1[0, 0] * p5_x + w1[1, 0] * F.interpolate(p6_td, scale_factor=2))
        p4_td = self.p4_td(w1[0, 1] * p4_x + w1[1, 1] * F.interpolate(p5_td, scale_factor=2))
        p3_td = self.p3_td(w1[0, 2] * p3_x + w1[1, 2] * F.interpolate(p4_td, scale_factor=2))
        p2_td = self.p2_td(w1[0, 3] * p2_x + w1[1, 3] * F.interpolate(p3_td, scale_factor=2))

        # Calculate Bottom-Up Pathway
        p2_out = p2_td
        p3_out = self.p3_out(w2[0, 0] * p3_x + w2[1, 0] * p3_td + w2[2, 0] * nn.Upsample(scale_factor=0.5)(p2_out))
        p4_out = self.p4_out(w2[0, 1] * p4_x + w2[1, 1] * p4_td + w2[2, 1] * nn.Upsample(scale_factor=0.5)(p3_out))
        p5_out = self.p5_out(w2[0, 2] * p5_x + w2[1, 2] * p5_td + w2[2, 2] * nn.Upsample(scale_factor=0.5)(p4_out))
        p6_out = self.p6_out(w2[0, 3] * p6_x + w2[1, 3] * p6_td + w2[2, 3] * nn.Upsample(scale_factor=0.5)(p5_out))

        return [p2_out, p3_out, p4_out, p5_out, p6_out]


class BiFPNDecoder(nn.Module):
    def __init__(self, num_filters=(64, 128, 256, 512), spatial_dims=3, out_channels=128, expansion=1, num_layers=2):
        super(BiFPNDecoder, self).__init__()
        if spatial_dims == 3:
            conv = nn.Conv3d
        else:
            conv = nn.Conv2d

        self.p2 = conv(num_filters[0] * expansion, out_channels, kernel_size=1, stride=1, padding=0)
        self.p3 = conv(num_filters[1] * expansion, out_channels, kernel_size=1, stride=1, padding=0)
        self.p4 = conv(num_filters[2] * expansion, out_channels, kernel_size=1, stride=1, padding=0)
        self.p5 = conv(num_filters[3] * expansion, out_channels, kernel_size=1, stride=1, padding=0)

        # p6 is computed by applying ReLU followed by a 3x3 stride-2 conv on p5
        self.p6 = ConvBlock(out_channels, out_channels, kernel_size=3, stride=2, padding=1, spatial_dims=spatial_dims)

        bifpns = []
        for _ in range(num_layers):
            bifpns.append(BiFPNBlock(out_channels, spatial_dims=spatial_dims))
        self.bifpn = nn.Sequential(*bifpns)

        self.spatial_dims = spatial_dims

    def forward(self, inputs, return_levels=False):
        c2, c3, c4, c5 = inputs['res2'], inputs['res3'], inputs['res4'], inputs['res5']

        # Calculate the input column of BiFPN
        p2_x = self.p2(c2)
        p3_x = self.p3(c3)
        p4_x = self.p4(c4)
        p5_x = self.p5(c5)
        p6_x = self.p6(p5_x)

        features = [p2_x, p3_x, p4_x, p5_x, p6_x]
        outputs = self.bifpn(features)
        if return_levels:
            return outputs[:-1]
        else:
            return outputs[0]

    def forward_predict(self, inputs):
        c2, c3, c4, c5 = inputs['res2'], inputs['res3'], inputs['res4'], inputs['res5']

        # Calculate the input column of BiFPN
        p2_x = self.p2(c2)
        p3_x = self.p3(c3)
        p4_x = self.p4(c4)
        p5_x = self.p5(c5)
        p6_x = self.p6(p5_x)

        features = [p2_x, p3_x, p4_x, p5_x, p6_x]
        return reversed(self.bifpn(features)[:-1])
