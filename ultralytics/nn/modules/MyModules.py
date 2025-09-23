import torch
from torch import nn


# EMA模块定义
class EMA(nn.Module):
    def __init__(self, channels, factor=8):
        super(EMA, self).__init__()
        self.groups = factor  # 定义分组数量
        assert channels // self.groups > 0  # 确保每组至少有一个通道
        self.softmax = nn.Softmax(-1)  # 定义Softmax层
        self.agp = nn.AdaptiveAvgPool2d((1, 1))  # 全局平均池化层
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))  # 沿宽度方向的平均池化
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))  # 沿高度方向的平均池化
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)  # 分组归一化层
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)  # 1x1卷积层
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)  # 3x3卷积层

    def forward(self, x):
        b, c, h, w = x.size()  # 获取输入的维度信息
        group_x = x.reshape(b * self.groups, -1, h, w)  # 对输入进行重塑，准备分组处理
        x_h = self.pool_h(group_x)  # 对每组进行宽度方向的池化
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)  # 对每组进行高度方向的池化并调整维度
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))  # 将池化后的特征沿特征维拼接并通过1x1卷积
        x_h, x_w = torch.split(hw, [h, w], dim=2)  # 分割特征以分别对应高和宽
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())  # 计算加权的特征并通过分组归一化
        x2 = self.conv3x3(group_x)  # 对分组的特征进行3x3卷积
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))  # 对x1进行全局池化和Softmax处理
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # 将x2重塑为三维张量以匹配x11
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))  # 对x2进行全局池化和Softmax处理
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # 将x1重塑为三维张量以匹配x21
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)  # 计算注意力权重
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)  # 将注意力权重应用于原始分组特征并重塑回原始维度



# 输入 N C HW,  输出 N C H W
if __name__ == '__main__':
    block = EMA(64).cuda()
    input = torch.rand(1, 64, 64, 64).cuda()
    output = block(input)
    print(input.size(), output.size())
