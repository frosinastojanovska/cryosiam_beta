import torch.nn as nn
import torch.nn.functional as F


class FPNDecoder(nn.Module):
    def __init__(self, num_filters=(64, 128, 256, 512), spatial_dims=3, out_channels=128,
                 expansion=1, skip_last_add=False, add_later_conv=False):
        super(FPNDecoder, self).__init__()
        if spatial_dims == 3:
            conv = nn.Conv3d
        else:
            conv = nn.Conv2d
        self.layer4 = conv(num_filters[0] * expansion, out_channels, kernel_size=1, stride=1, padding=0)
        self.layer3 = conv(num_filters[1] * expansion, out_channels, kernel_size=1, stride=1, padding=0)
        self.layer2 = conv(num_filters[2] * expansion, out_channels, kernel_size=1, stride=1, padding=0)
        self.layer1 = conv(num_filters[3] * expansion, out_channels, kernel_size=1, stride=1, padding=0)

        self.add_later_conv = add_later_conv
        if add_later_conv:
            self.later_conv1 = conv(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
            self.later_conv2 = conv(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
            self.later_conv3 = conv(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
            self.later_conv4 = conv(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self.skip_last_add = skip_last_add
        self.spatial_dims = spatial_dims

    def forward(self, x, return_levels=False):
        o1 = self.layer1(x['res5'])
        o2 = self.upsample_add(o1, self.layer2(x['res4']))
        o3 = self.upsample_add(o2, self.layer3(x['res3']))
        if self.skip_last_add:
            o4 = self.upsample_add(o3, self.layer4(x['res2']), True)
        else:
            o4 = self.upsample_add(o3, self.layer4(x['res2']))

        if self.add_later_conv:
            o1 = self.later_conv1(o1)
            o2 = self.later_conv2(o2)
            o3 = self.later_conv3(o3)
            o4 = self.later_conv4(o4)
        if return_levels:
            return o4, o3, o2, o1
        else:
            return o4

    def forward_predict(self, x):
        o1 = self.layer1(x['res5'])
        o2 = self.upsample_add(o1, self.layer2(x['res4']))
        o3 = self.upsample_add(o2, self.layer3(x['res3']))
        if self.skip_last_add:
            o4 = self.upsample_add(o3, self.layer4(x['res2']), True)
        else:
            o4 = self.upsample_add(o3, self.layer4(x['res2']))

        if self.add_later_conv:
            o1 = self.later_conv1(o1)
            o2 = self.later_conv2(o2)
            o3 = self.later_conv3(o3)
            o4 = self.later_conv4(o4)
        return o1, o2, o3, o4

    def upsample_add(self, x, y, no_add=False):
        if self.spatial_dims == 3:
            _, _, D, H, W = y.size()
            if no_add:
                return F.interpolate(x, size=(D, H, W), mode='trilinear', align_corners=False)
            else:
                return F.interpolate(x, size=(D, H, W), mode='trilinear', align_corners=False) + y
        else:
            _, _, H, W = y.size()
            if no_add:
                return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)
            else:
                return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False) + y
