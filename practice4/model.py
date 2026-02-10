import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class REBNCONV3d(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, dirate=1):
        super(REBNCONV3d, self).__init__()
        self.conv_s1 = nn.Conv3d(in_ch, out_ch, 3, padding=1 * dirate, dilation=1 * dirate)
        self.bn_s1 = nn.BatchNorm3d(out_ch)
        self.relu_s1 = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu_s1(self.bn_s1(self.conv_s1(x)))

def _upsample_like(src, tar):
    src = F.interpolate(src, size=tar.shape[2:], mode='trilinear', align_corners=True)
    return src

### RSU Block 3D ###
class RSU7_3d(nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU7_3d, self).__init__()
        self.rebnconvin = REBNCONV3d(in_ch, out_ch, dirate=1)
        self.rebnconv1 = REBNCONV3d(out_ch, mid_ch, dirate=1)
        self.rebnconv2 = REBNCONV3d(mid_ch, mid_ch, dirate=1)
        self.rebnconv3 = REBNCONV3d(mid_ch, mid_ch, dirate=1)
        self.rebnconv4 = REBNCONV3d(mid_ch, mid_ch, dirate=1)
        self.rebnconv5 = REBNCONV3d(mid_ch, mid_ch, dirate=1)
        self.rebnconv6 = REBNCONV3d(mid_ch, mid_ch, dirate=1)
        self.rebnconv7 = REBNCONV3d(mid_ch, mid_ch, dirate=2)

        self.rebnconv6d = REBNCONV3d(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv5d = REBNCONV3d(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv4d = REBNCONV3d(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv3d = REBNCONV3d(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV3d(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV3d(mid_ch * 2, out_ch, dirate=1)

        self.pool = nn.MaxPool3d(2, stride=2, ceil_mode=True)

    def forward(self, x):
        hx = x
        hxin = self.rebnconvin(hx)
        hx1 = self.rebnconv1(hxin)
        hx = self.pool(hx1)
        hx2 = self.rebnconv2(hx)
        hx = self.pool(hx2)
        hx3 = self.rebnconv3(hx)
        hx = self.pool(hx3)
        hx4 = self.rebnconv4(hx)
        hx = self.pool(hx4)
        hx5 = self.rebnconv5(hx)
        hx = self.pool(hx5)
        hx6 = self.rebnconv6(hx)
        hx7 = self.rebnconv7(hx6)

        hx6d = self.rebnconv6d(torch.cat((hx7, hx6), 1))
        hx6dup = _upsample_like(hx6d, hx5)
        hx5d = self.rebnconv5d(torch.cat((hx6dup, hx5), 1))
        hx5dup = _upsample_like(hx5d, hx4)
        hx4d = self.rebnconv4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)
        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)
        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)
        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))
        return hx1d + hxin

class RSU6_3d(nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU6_3d, self).__init__()
        self.rebnconvin = REBNCONV3d(in_ch, out_ch, dirate=1)
        self.rebnconv1 = REBNCONV3d(out_ch, mid_ch, dirate=1)
        self.rebnconv2 = REBNCONV3d(mid_ch, mid_ch, dirate=1)
        self.rebnconv3 = REBNCONV3d(mid_ch, mid_ch, dirate=1)
        self.rebnconv4 = REBNCONV3d(mid_ch, mid_ch, dirate=1)
        self.rebnconv5 = REBNCONV3d(mid_ch, mid_ch, dirate=1)
        self.rebnconv6 = REBNCONV3d(mid_ch, mid_ch, dirate=2)

        self.rebnconv5d = REBNCONV3d(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv4d = REBNCONV3d(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv3d = REBNCONV3d(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV3d(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV3d(mid_ch * 2, out_ch, dirate=1)

        self.pool = nn.MaxPool3d(2, stride=2, ceil_mode=True)

    def forward(self, x):
        hx = x
        hxin = self.rebnconvin(hx)
        hx1 = self.rebnconv1(hxin)
        hx = self.pool(hx1)
        hx2 = self.rebnconv2(hx)
        hx = self.pool(hx2)
        hx3 = self.rebnconv3(hx)
        hx = self.pool(hx3)
        hx4 = self.rebnconv4(hx)
        hx = self.pool(hx4)
        hx5 = self.rebnconv5(hx)
        hx6 = self.rebnconv6(hx5)

        hx5d = self.rebnconv5d(torch.cat((hx6, hx5), 1))
        hx5dup = _upsample_like(hx5d, hx4)
        hx4d = self.rebnconv4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)
        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)
        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)
        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))
        return hx1d + hxin

class RSU5_3d(nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU5_3d, self).__init__()
        self.rebnconvin = REBNCONV3d(in_ch, out_ch, dirate=1)
        self.rebnconv1 = REBNCONV3d(out_ch, mid_ch, dirate=1)
        self.rebnconv2 = REBNCONV3d(mid_ch, mid_ch, dirate=1)
        self.rebnconv3 = REBNCONV3d(mid_ch, mid_ch, dirate=1)
        self.rebnconv4 = REBNCONV3d(mid_ch, mid_ch, dirate=1)
        self.rebnconv5 = REBNCONV3d(mid_ch, mid_ch, dirate=2)

        self.rebnconv4d = REBNCONV3d(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv3d = REBNCONV3d(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV3d(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV3d(mid_ch * 2, out_ch, dirate=1)

        self.pool = nn.MaxPool3d(2, stride=2, ceil_mode=True)

    def forward(self, x):
        hx = x
        hxin = self.rebnconvin(hx)
        hx1 = self.rebnconv1(hxin)
        hx = self.pool(hx1)
        hx2 = self.rebnconv2(hx)
        hx = self.pool(hx2)
        hx3 = self.rebnconv3(hx)
        hx = self.pool(hx3)
        hx4 = self.rebnconv4(hx)
        hx5 = self.rebnconv5(hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)
        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)
        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)
        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))
        return hx1d + hxin

class RSU4_3d(nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU4_3d, self).__init__()
        self.rebnconvin = REBNCONV3d(in_ch, out_ch, dirate=1)
        self.rebnconv1 = REBNCONV3d(out_ch, mid_ch, dirate=1)
        self.rebnconv2 = REBNCONV3d(mid_ch, mid_ch, dirate=1)
        self.rebnconv3 = REBNCONV3d(mid_ch, mid_ch, dirate=1)
        self.rebnconv4 = REBNCONV3d(mid_ch, mid_ch, dirate=2)

        self.rebnconv3d = REBNCONV3d(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV3d(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV3d(mid_ch * 2, out_ch, dirate=1)
        self.pool = nn.MaxPool3d(2, stride=2, ceil_mode=True)

    def forward(self, x):
        hx = x
        hxin = self.rebnconvin(hx)
        hx1 = self.rebnconv1(hxin)
        hx = self.pool(hx1)
        hx2 = self.rebnconv2(hx)
        hx = self.pool(hx2)
        hx3 = self.rebnconv3(hx)
        hx4 = self.rebnconv4(hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)
        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)
        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))
        return hx1d + hxin

class RSU4F_3d(nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU4F_3d, self).__init__()
        self.rebnconvin = REBNCONV3d(in_ch, out_ch, dirate=1)
        self.rebnconv1 = REBNCONV3d(out_ch, mid_ch, dirate=1)
        self.rebnconv2 = REBNCONV3d(mid_ch, mid_ch, dirate=2)
        self.rebnconv3 = REBNCONV3d(mid_ch, mid_ch, dirate=4)
        self.rebnconv4 = REBNCONV3d(mid_ch, mid_ch, dirate=8)

        self.rebnconv3d = REBNCONV3d(mid_ch * 2, mid_ch, dirate=4)
        self.rebnconv2d = REBNCONV3d(mid_ch * 2, mid_ch, dirate=2)
        self.rebnconv1d = REBNCONV3d(mid_ch * 2, out_ch, dirate=1)

    def forward(self, x):
        hxin = self.rebnconvin(x)
        hx1 = self.rebnconv1(hxin)
        hx2 = self.rebnconv2(hx1)
        hx3 = self.rebnconv3(hx2)
        hx4 = self.rebnconv4(hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4, hx3), 1))
        hx2d = self.rebnconv2d(torch.cat((hx3d, hx2), 1))
        hx1d = self.rebnconv1d(torch.cat((hx2d, hx1), 1))
        return hx1d + hxin

class RegressionHead(nn.Module):
    def __init__(self, in_features, num_anchors=9, feature_size=256):
        super(RegressionHead, self).__init__()
        self.conv1 = nn.Conv3d(in_features, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv3d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()
        self.conv3 = nn.Conv3d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()
        self.conv4 = nn.Conv3d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()
        
        # 6 coordinates for 3D box: [x, y, z, w, h, d]
        self.output = nn.Conv3d(feature_size, num_anchors * 6, kernel_size=3, padding=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)
        out = self.conv2(out)
        out = self.act2(out)
        out = self.conv3(out)
        out = self.act3(out)
        out = self.conv4(out)
        out = self.act4(out)
        out = self.output(out)
        return out

class ClassificationHead(nn.Module):
    def __init__(self, in_features, num_anchors=9, num_classes=1, feature_size=256):
        super(ClassificationHead, self).__init__()
        self.conv1 = nn.Conv3d(in_features, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv3d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()
        self.conv3 = nn.Conv3d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()
        self.conv4 = nn.Conv3d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()
        self.output = nn.Conv3d(feature_size, num_anchors * num_classes, kernel_size=3, padding=1)
        self.output_act = nn.Sigmoid()

    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)
        out = self.conv2(out)
        out = self.act2(out)
        out = self.conv3(out)
        out = self.act3(out)
        out = self.conv4(out)
        out = self.act4(out)
        out = self.output(out)
        out = self.output_act(out)
        return out

class RetinaU2NET3d(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, num_classes=1, num_anchors=1):
        super(RetinaU2NET3d, self).__init__()
        # Encoder
        self.stage1 = RSU7_3d(in_ch, 16, 64)
        self.pool12 = nn.MaxPool3d(2, stride=2, ceil_mode=True)

        self.stage2 = RSU6_3d(64, 16, 64)
        self.pool23 = nn.MaxPool3d(2, stride=2, ceil_mode=True)

        self.stage3 = RSU5_3d(64, 16, 64)
        self.pool34 = nn.MaxPool3d(2, stride=2, ceil_mode=True)

        self.stage4 = RSU4_3d(64, 16, 64)
        self.pool45 = nn.MaxPool3d(2, stride=2, ceil_mode=True)

        self.stage5 = RSU4F_3d(64, 16, 64)
        self.pool56 = nn.MaxPool3d(2, stride=2, ceil_mode=True)

        self.stage6 = RSU4F_3d(64, 16, 64)

        # Decoder
        self.stage5d = RSU4F_3d(128, 16, 64)
        self.stage4d = RSU4_3d(128, 16, 64)
        self.stage3d = RSU5_3d(128, 16, 64)
        self.stage2d = RSU6_3d(128, 16, 64)
        self.stage1d = RSU7_3d(128, 16, 64)

        self.side1 = nn.Conv3d(64, out_ch, 3, padding=1)
        self.side2 = nn.Conv3d(64, out_ch, 3, padding=1)
        self.side3 = nn.Conv3d(64, out_ch, 3, padding=1)
        self.side4 = nn.Conv3d(64, out_ch, 3, padding=1)
        self.side5 = nn.Conv3d(64, out_ch, 3, padding=1)
        self.side6 = nn.Conv3d(64, out_ch, 3, padding=1)

        self.outconv = nn.Conv3d(6, out_ch, 1) # fusion

        # Retina Heads - Attach them to Decoder stages (Feature Pyramid)
        # Stages 3, 4, 5, 6
        self.regression_head = RegressionHead(in_features=64, num_anchors=num_anchors, feature_size=64)
        self.classification_head = ClassificationHead(in_features=64, num_anchors=num_anchors, num_classes=num_classes, feature_size=64)

    def forward(self, x):
        hx = x
        
        # Encoder
        hx1 = self.stage1(hx)
        hx = self.pool12(hx1)

        hx2 = self.stage2(hx)
        hx = self.pool23(hx2)

        hx3 = self.stage3(hx)
        hx = self.pool34(hx3)

        hx4 = self.stage4(hx)
        hx = self.pool45(hx4)

        hx5 = self.stage5(hx)
        hx = self.pool56(hx5)

        hx6 = self.stage6(hx)
        
        # Decoder
        hx6up = _upsample_like(hx6, hx5)
        hx5d = self.stage5d(torch.cat((hx6up, hx5), 1))
        
        hx5dup = _upsample_like(hx5d, hx4)
        hx4d = self.stage4d(torch.cat((hx5dup, hx4), 1))
        
        hx4dup = _upsample_like(hx4d, hx3)
        hx3d = self.stage3d(torch.cat((hx4dup, hx3), 1))
        
        hx3dup = _upsample_like(hx3d, hx2)
        hx2d = self.stage2d(torch.cat((hx3dup, hx2), 1))
        
        hx2dup = _upsample_like(hx2d, hx1)
        hx1d = self.stage1d(torch.cat((hx2dup, hx1), 1))

        # Side Outputs (Segmentation Maps)
        d1 = self.side1(hx1d)
        d2 = self.side2(hx2d)
        d3 = self.side3(hx3d)
        d4 = self.side4(hx4d)
        d5 = self.side5(hx5d)
        d6 = self.side6(hx6)

        d2 = _upsample_like(d2, d1)
        d3 = _upsample_like(d3, d1)
        d4 = _upsample_like(d4, d1)
        d5 = _upsample_like(d5, d1)
        d6 = _upsample_like(d6, d1)

        d0 = self.outconv(torch.cat((d1, d2, d3, d4, d5, d6), 1))
        
        # Retina Detection Heads (Class + BBox)
        # Applied to multi-scale features: hx3d, hx4d, hx5d, hx6 (Deepest)
        # Similar to FPN levels P3, P4, P5, P6
        
        features = [hx3d, hx4d, hx5d, hx6]
        
        cls_preds = []
        reg_preds = []
        
        for feature in features:
            cls_preds.append(self.classification_head(feature))
            reg_preds.append(self.regression_head(feature))

        # Returns: 
        #   masks: list of probability maps [d0, d1, d2, d3, d4, d5, d6] (d0 is fused)
        #   cls: list of classification maps
        #   bbox: list of regression maps
        
        return [d0, d1, d2, d3, d4, d5, d6], cls_preds, reg_preds
