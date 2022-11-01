import torch
import torch.nn as nn
import torch.nn.functional as F
from util import sample_and_group

class Local_op(nn.Module):
    def __init__(self, in_channels, out_channels): # [128,128] [256, 256]
        super(Local_op, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        b, n, s, d = x.size()  # torch.Size([32, 512/256, 32, 128/256]) [batch npoint sample d]
        x = x.permute(0, 1, 3, 2) # [batch npoint d sample]
        x = x.reshape(-1, d, s) # [512*32, 128, 32] [256*32, 256, 32] [batch channel npoint]
        batch_size, _, N = x.size()
        x = F.relu(self.bn1(self.conv1(x)))  # B, D, N
        x = F.relu(self.bn2(self.conv2(x)))  # B, D, N [512*32, 128, 32]
        x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1) # x->[512*32 128 32] -> [512*32 128 1] -> torch.Size([512*32, 128])
        x = x.reshape(b, n, -1).permute(0, 2, 1) # [512*32 128] -> [32 512 128] -> [32 128 512]
        # print('x_2', x.shape)
        return x

class Point_Transformer_Last(nn.Module):
    def __init__(self, args, channels=256):
        super(Point_Transformer_Last, self).__init__()
        self.args = args
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)

        self.bn1 = nn.BatchNorm1d(channels)
        self.bn2 = nn.BatchNorm1d(channels)

        self.sa1 = SA_Layer(channels)
        self.sa2 = SA_Layer(channels)
        self.sa3 = SA_Layer(channels)
        self.sa4 = SA_Layer(channels)

    def forward(self, x):
        batch_size, _, N = x.size()

        # B, D, N
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x1 = self.sa1(x)
        x2 = self.sa2(x1)
        x3 = self.sa3(x2)
        x4 = self.sa4(x3)
        x = torch.cat((x1, x2, x3, x4), dim=1) # 256*4

        return x


class SA_Layer(nn.Module):
    def __init__(self, channels):
        super(SA_Layer, self).__init__()
        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight
        self.q_conv.bias = self.k_conv.bias

        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # b, n, c
        x_q = self.q_conv(x).permute(0, 2, 1)
        # b, c, n
        x_k = self.k_conv(x)
        x_v = self.v_conv(x)
        # b, n, n
        energy = torch.bmm(x_q, x_k)

        attention = self.softmax(energy)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdim=True))
        # b, c, n
        x_r = torch.bmm(x_v, attention)
        x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
        x = x + x_r
        return x


class SPct(nn.Module):
    #  SPCT, with point embedding and offset-attention
    def __init__(self, args, output_channels=40):
        super(SPct, self).__init__()
        self.args = args
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 256, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(256)

        self.pt_last = Point_Transformer_Last(args) # 256

        self.conv_fuse = nn.Sequential(nn.Conv1d(1024, 1024, kernel_size=1, bias=False),
                                       nn.BatchNorm1d(1024),
                                       nn.LeakyReLU(negative_slope=0.2))

        self.linear1 = nn.Linear(1024, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x):
        # LBR
        xyz = x.permute(0, 2, 1)  # [B 3 npoints]
        batch_size, _, _ = x.size()
        x = F.relu(self.bn1(self.conv1(x)))  # [B 64 npoints]
        # B, D, N
        x = F.relu(self.bn2(self.conv2(x)))  # [B 64 npoints]
        # x = x.permute(0, 2, 1)  # [B npoints 64]
        # print('x_1', x.shape)
        # SG -> 最远点采样(512) + KNN(32)
        # 先从原始的点云数据中采取512个点 knn32
        # x = self.pt_last(feature_1) # x torch.Size([32, 1024, 256])
        # x = torch.cat([x, feature_1], dim=1)  # 多出的？？？ #  torch.Size([32, 1280, 256])
        x = self.pt_last(x)  #  torch.Size([32, 1024, 256])
        x = self.conv_fuse(x)
        x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        return x
