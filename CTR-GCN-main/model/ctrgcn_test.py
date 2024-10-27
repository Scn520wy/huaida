import math
import pdb

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def conv_branch_init(conv, branches):
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal_(weight, 0, math.sqrt(2. / (n * k1 * k2 * branches)))
    nn.init.constant_(conv.bias, 0)


def conv_init(conv):
    if conv.weight is not None:
        nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        if hasattr(m, 'weight'):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
        if hasattr(m, 'bias') and m.bias is not None and isinstance(m.bias, torch.Tensor):
            nn.init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            m.weight.data.normal_(1.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            m.bias.data.fill_(0)


class TemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super(TemporalConv, self).__init__()
        pad = (kernel_size + (kernel_size-1) * (dilation-1) - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=(pad, 0),
            stride=(stride, 1),
            dilation=(dilation, 1))

        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class MultiScale_TemporalConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 dilations=[1,2,3,4],
                 residual=True,
                 residual_kernel_size=1):

        super().__init__()
        assert out_channels % (len(dilations) + 2) == 0, '# out channels should be multiples of # branches'

        # Multiple branches of temporal convolution
        self.num_branches = len(dilations) + 2
        branch_channels = out_channels // self.num_branches
        if type(kernel_size) == list:
            assert len(kernel_size) == len(dilations)
        else:
            kernel_size = [kernel_size]*len(dilations)
        # Temporal Convolution branches
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    branch_channels,
                    kernel_size=1,
                    padding=0),
                nn.BatchNorm2d(branch_channels),
                nn.ReLU(inplace=True),
                TemporalConv(
                    branch_channels,
                    branch_channels,
                    kernel_size=ks,
                    stride=stride,
                    dilation=dilation),
            )
            for ks, dilation in zip(kernel_size, dilations)
        ])

        # Additional Max & 1x1 branch
        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3,1), stride=(stride,1), padding=(1,0)),
            nn.BatchNorm2d(branch_channels)  # 为什么还要加bn
        ))

        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0, stride=(stride,1)),
            nn.BatchNorm2d(branch_channels)
        ))

        # Residual connection
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = TemporalConv(in_channels, out_channels, kernel_size=residual_kernel_size, stride=stride)

        # initialize
        self.apply(weights_init)

    def forward(self, x):
        # Input dim: (N,C,T,V)
        res = self.residual(x)
        branch_outs = []
        for tempconv in self.branches:
            out = tempconv(x)
            branch_outs.append(out)

        out = torch.cat(branch_outs, dim=1)
        out += res
        return out


class CTRGC(nn.Module):
    def __init__(self, in_channels, out_channels, rel_reduction=8, mid_reduction=1):
        super(CTRGC, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if in_channels == 3 or in_channels == 9:
            self.rel_channels = 8
            self.mid_channels = 16
        else:
            self.rel_channels = in_channels // rel_reduction
            self.mid_channels = in_channels // mid_reduction
        self.conv1 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1)
        self.conv4 = nn.Conv2d(self.rel_channels, self.out_channels, kernel_size=1)
        self.tanh = nn.Tanh()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)

    def forward(self, x, A=None, alpha=1):
        x1, x2, x3 = self.conv1(x).mean(-2), self.conv2(x).mean(-2), self.conv3(x)
        x1 = self.tanh(x1.unsqueeze(-1) - x2.unsqueeze(-2))
        x1 = self.conv4(x1) * alpha + (A.unsqueeze(0).unsqueeze(0) if A is not None else 0)  # N,C,V,V
        x1 = torch.einsum('ncuv,nctv->nctu', x1, x3)
        return x1

class unit_tcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super(unit_tcn, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
                              stride=(stride, 1))

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x


class unit_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, A, coff_embedding=4, adaptive=True, residual=True):
        super(unit_gcn, self).__init__()
        inter_channels = out_channels // coff_embedding
        self.inter_c = inter_channels
        self.out_c = out_channels
        self.in_c = in_channels
        self.adaptive = adaptive
        self.num_subset = A.shape[0]
        self.convs = nn.ModuleList()
        for i in range(self.num_subset):
            self.convs.append(CTRGC(in_channels, out_channels))

        if residual:
            if in_channels != out_channels:
                self.down = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1),
                    nn.BatchNorm2d(out_channels)
                )
            else:
                self.down = lambda x: x
        else:
            self.down = lambda x: 0
        if self.adaptive:
            self.PA = nn.Parameter(torch.from_numpy(A.astype(np.float32)))
        else:
            self.A = Variable(torch.from_numpy(A.astype(np.float32)), requires_grad=False)
        self.alpha = nn.Parameter(torch.zeros(1))
        self.bn = nn.BatchNorm2d(out_channels)
        self.soft = nn.Softmax(-2)
        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)

    def forward(self, x):
        y = None
        if self.adaptive:
            A = self.PA
        else:
            A = self.A.cuda(x.get_device())
        for i in range(self.num_subset):
            z = self.convs[i](x, A[i], self.alpha)
            y = z + y if y is not None else z
        y = self.bn(y)
        y += self.down(x)
        y = self.relu(y)


        return y


class TCN_GCN_unit(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True, adaptive=True, kernel_size=5, dilations=[1,2]):
        super(TCN_GCN_unit, self).__init__()
        self.gcn1 = unit_gcn(in_channels, out_channels, A, adaptive=adaptive)
        self.tcn1 = MultiScale_TemporalConv(out_channels, out_channels, kernel_size=kernel_size, stride=stride, dilations=dilations,
                                            residual=False)
        self.relu = nn.ReLU(inplace=True)
        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        y = self.relu(self.tcn1(self.gcn1(x)) + self.residual(x))
        return y


class Spa_Atten(nn.Module):
    def __init__(self, out_channels):
        super(Spa_Atten, self).__init__()
        self.out_channels = out_channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.relu = nn.ReLU()
        self.soft = nn.Softmax(-1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.linear = nn.Linear(17, 17)

    #        self.linear2 = nn.Linear(25,25)

    def forward(self, x):
        N, C, T, V = x.size()
        x1 = x[:, :C // 2, :, :]
        x2 = x[:, C // 2:C, :, :]
        Q_o = Q_Spa_Trans = self.avg_pool(x1.permute(0, 3, 1, 2).contiguous())
        K_o = K_Spa_Trans = self.avg_pool(x2.permute(0, 3, 1, 2).contiguous())
        Q_Spa_Trans = self.relu(self.linear(Q_Spa_Trans.squeeze(-1).squeeze(-1)))
        K_Spa_Trans = self.relu(self.linear(K_Spa_Trans.squeeze(-1).squeeze(-1)))
        Spa_atten = self.soft(torch.einsum('nv,nw->nvw', (Q_Spa_Trans, K_Spa_Trans))).unsqueeze(1).repeat(1,
                                                                                                          self.out_channels,
                                                                                                          1, 1)
        return Spa_atten, Q_o, K_o


class Spatial_MixFormer(nn.Module):
    def __init__(self, in_channels, out_channels, A, groups=8, coff_embedding=4, num_subset=3, t_stride=1, t_padding=0,
                 t_dilation=1, bias=True, first=False, residual=True):
        super(Spatial_MixFormer, self).__init__()
        inter_channels = out_channels // coff_embedding
        self.inter_c = inter_channels
        self.groups = groups
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.num_subset = 3
        self.alpha = nn.Parameter(torch.ones(1))
        self.A_GEME = nn.Parameter(torch.tensor(np.reshape(A.astype(np.float32), [3, 1, 17, 17]), dtype=torch.float32,
                                                requires_grad=True).repeat(1, groups, 1, 1), requires_grad=True)
        self.A_SE = Variable(torch.from_numpy(np.reshape(A.astype(np.float32), [3, 1, 17, 17]).repeat(groups, axis=1)),
                             requires_grad=False)
        self.sigmoid = nn.Sigmoid()
        self.linear = nn.Linear(17, 17)
        self.Spa_Att = Spa_Atten(out_channels // 4)
        self.AvpChaRef = nn.AdaptiveAvgPool2d(1)
        self.ChaRef_conv = nn.Conv1d(1, 1, kernel_size=3, padding=(3 - 1) // 2, bias=False)
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * num_subset,
            kernel_size=(1, 1),
            padding=(t_padding, 0),
            stride=(t_stride, 1),
            dilation=(t_dilation, 1),
            bias=bias)
        if residual:
            if in_channels != out_channels:
                self.down = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1),
                    nn.BatchNorm2d(out_channels)
                )
            else:
                self.down = lambda x: x
        else:
            self.down = lambda x: 0
        # self.fc = nn.Linear(50, 25)
        # nn.init.normal(self.fc.weight, 0, math.sqrt(2. / 25))
        self.bn = nn.BatchNorm2d(out_channels)
        self.soft = nn.Softmax(-1)
        self.relu = nn.ReLU()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)

    def forward(self, x0):

        N, C, T, V = x0.size()
        A = self.A_SE.cuda(x0.get_device()) + self.A_GEME
        norm_learn_A = A.repeat(1, self.out_channels // self.groups, 1, 1)
        A_final = torch.zeros([N, self.num_subset, self.out_channels, 17, 17], dtype=torch.float,
                              device=x0.get_device()).detach()
        m = x0
        m = self.conv(m)
        n, kc, t, v = m.size()
        m = m.view(n, self.num_subset, kc // self.num_subset, t, v)
        for i in range(self.num_subset):
            m1, Q1, K1 = self.Spa_Att(m[:, i, :(kc // self.num_subset) // 4, :, :])
            m2, Q2, K2 = self.Spa_Att(m[:, i, (kc // self.num_subset) // 4:((kc // self.num_subset) // 4) * 2, :, :])
            m3, Q3, K3 = self.Spa_Att(
                m[:, i, ((kc // self.num_subset) // 4) * 2:((kc // self.num_subset) // 4) * 3, :, :])
            m4, Q4, K4 = self.Spa_Att(
                m[:, i, ((kc // self.num_subset) // 4) * 3:((kc // self.num_subset) // 4) * 4, :, :])
            m1_2 = self.soft(torch.einsum('nv,nw->nvw', (self.relu(self.linear(K1.squeeze(-1).squeeze(-1))), self.relu(
                self.linear(Q2.squeeze(-1).squeeze(-1)))))).unsqueeze(1).repeat(1, self.out_channels // 4, 1, 1)
            m2_3 = self.soft(torch.einsum('nv,nw->nvw', (self.relu(self.linear(K2.squeeze(-1).squeeze(-1))), self.relu(
                self.linear(Q3.squeeze(-1).squeeze(-1)))))).unsqueeze(1).repeat(1, self.out_channels // 4, 1, 1)
            m3_4 = self.soft(torch.einsum('nv,nw->nvw', (self.relu(self.linear(K3.squeeze(-1).squeeze(-1))), self.relu(
                self.linear(Q4.squeeze(-1).squeeze(-1)))))).unsqueeze(1).repeat(1, self.out_channels // 4, 1, 1)

            m1 = m1 / 2 + m1_2 / 4
            m2 = m2 / 2 + m1_2 / 4 + m2_3 / 4
            m3 = m3 / 2 + m2_3 / 4 + m3_4 / 4
            m4 = m4 / 2 + m3_4 / 4
            atten = torch.cat([m1, m2, m3, m4], dim=1)
            A_final[:, i, :, :, :] = atten * 0.5 + norm_learn_A[i]
        m = torch.einsum('nkctv,nkcvw->nctw', (m, A_final))

        # Channel Reforming
        CR_in = self.AvpChaRef(m)
        CR_in = self.ChaRef_conv(CR_in.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        CR_out = m + m * self.sigmoid(CR_in).expand_as(m)

        out = self.bn(CR_out)
        out += self.down(x0)
        out = self.relu(out)
        return out


class Tem_Seq_h(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(Tem_Seq_h, self).__init__()
        pad = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
                              stride=(stride, 1))
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        conv_init(self.conv)
        bn_init(self.bn, 1)
        self.AvpTemSeq = nn.AdaptiveAvgPool2d(1)
        self.MaxTemSeq = nn.AdaptiveMaxPool2d(1)
        self.combine_conv = nn.Conv2d(2, 1, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv(x)
        N, C, T, V = x.size()
        x = x.permute(0, 2, 1, 3).contiguous()
        Q_Tem_Seq = self.AvpTemSeq(x)
        K_Tem_Seq = self.MaxTemSeq(x)
        Combine = torch.cat([Q_Tem_Seq, K_Tem_Seq], dim=2)
        Combine = self.combine_conv(Combine.permute(0, 2, 1, 3).contiguous()).permute(0, 2, 1, 3).contiguous()
        Tem_Seq_out = (x * self.sigmoid(Combine).expand_as(x)).permute(0, 2, 1, 3).contiguous()
        return Tem_Seq_out


class Tem_Trans(nn.Module):
    def __init__(self, in_channels, out_channels, Frames, kernel_size, stride=1):
        super(Tem_Trans, self).__init__()
        pad = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
                              stride=(stride, 1))
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.AvpTemTrans = nn.AdaptiveAvgPool2d(1)
        self.MaxTemTrans = nn.AdaptiveMaxPool2d(1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.soft = nn.Softmax(-1)
        self.linear = nn.Linear(Frames, Frames)

    def forward(self, x):
        x = self.conv(x)
        N, C, T, V = x.size()
        x1 = x[:, :C // 2, :, :]
        x2 = x[:, C // 2:C, :, :]
        Q_Tem_Trans = self.AvpTemTrans(x1.permute(0, 2, 1, 3).contiguous())
        K_Tem_Trans = self.MaxTemTrans(x2.permute(0, 2, 1, 3).contiguous())
        Q_Tem_Trans = self.relu(self.linear(Q_Tem_Trans.squeeze(-1).squeeze(-1)))
        K_Tem_Trans = self.relu(self.linear(K_Tem_Trans.squeeze(-1).squeeze(-1)))
        Tem_atten = self.sigmoid(torch.einsum('nt,nm->ntm', (Q_Tem_Trans, K_Tem_Trans)))
        Tem_Trans_out = self.bn(torch.einsum('nctv,ntm->ncmv', (x, Tem_atten)))
        return Tem_Trans_out


class TemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super(TemporalConv, self).__init__()
        pad = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=(pad, 0),
            stride=(stride, 1),
            dilation=(dilation, 1))
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class Temporal_MixFormer(nn.Module):
    def __init__(self, in_channels, out_channels, Frames, kernel_size=3, stride=1, dilations=[1, 2, 3, 4],
                 residual=True, residual_kernel_size=1):
        super().__init__()
        assert out_channels % (len(dilations) + 3) == 0
        self.num_branches = len(dilations) + 3
        branch_channels = out_channels // self.num_branches
        if type(kernel_size) == list:
            assert len(kernel_size) == len(dilations)
        else:
            kernel_size = [kernel_size] * len(dilations)
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0), nn.BatchNorm2d(branch_channels),
                nn.ReLU(inplace=True),
                TemporalConv(branch_channels, branch_channels, kernel_size=ks, stride=stride, dilation=dilation), )
            for ks, dilation in zip(kernel_size, dilations)
        ])
        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 1), stride=(stride, 1), padding=(1, 0)),
            nn.BatchNorm2d(branch_channels)
        ))
        self.branches.append(nn.Sequential(
            Tem_Trans(in_channels, branch_channels, Frames, kernel_size=1, stride=stride),
            nn.BatchNorm2d(branch_channels)
        ))
        self.branches.append(nn.Sequential(
            Tem_Seq_h(in_channels, branch_channels, kernel_size=1, stride=stride),
            nn.BatchNorm2d(branch_channels)
        ))
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = TemporalConv(in_channels, out_channels, kernel_size=residual_kernel_size, stride=stride)
        else:
            self.residual = TemporalConv(in_channels, out_channels, kernel_size=residual_kernel_size, stride=stride)
        self.apply(weights_init)

    def forward(self, x):
        res = self.residual(x)
        branch_outs = []
        for tempconv in self.branches:
            out = tempconv(x)
            branch_outs.append(out)
        out = torch.cat(branch_outs, dim=1)
        out += res
        return out

class unit_skip(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super(unit_skip, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),stride=(stride, 1))
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x

class Ske_MixF(nn.Module):
    def __init__(self, in_channels, out_channels, A, Frames, stride=1, residual=True):
        super(Ske_MixF, self).__init__()
        self.spa_mixf = Spatial_MixFormer(in_channels, out_channels, A)
        self.tem_mixf = Temporal_MixFormer(out_channels, out_channels, Frames, kernel_size=5, stride=stride, dilations=[1],residual=False)
        self.relu = nn.ReLU()
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = unit_skip(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        x = self.tem_mixf(self.spa_mixf(x)) + self.residual(x)
        return self.relu(x)



class Model(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_person=2, graph=None, graph_args=dict(), in_channels=3,
                 drop_out=0, adaptive=True):
        super(Model, self).__init__()

        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)

        A = self.graph.A # 3,25,25

        self.num_class = num_class
        self.num_point = num_point
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        base_channel = 64
        self.l1 = TCN_GCN_unit(in_channels, base_channel, A, residual=False, adaptive=adaptive)
        self.l2 = TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive)
        self.l3 = TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive)
        self.l4 = TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive)
        self.l5 = TCN_GCN_unit(base_channel, base_channel*2, A, stride=2, adaptive=adaptive)
        self.l6 = TCN_GCN_unit(base_channel*2, base_channel*2, A, adaptive=adaptive)
        self.l7 = TCN_GCN_unit(base_channel*2, base_channel*2, A, adaptive=adaptive)
        self.l8 = TCN_GCN_unit(base_channel*2, base_channel*4, A, stride=2, adaptive=adaptive)
        self.l9 = TCN_GCN_unit(base_channel*4, base_channel*4, A, adaptive=adaptive)
        self.l10 = TCN_GCN_unit(base_channel*4, base_channel*4, A, adaptive=adaptive)

        self.m1 = Ske_MixF(in_channels, base_channel, A, 64, residual=False)
        self.m2 = Ske_MixF(base_channel, base_channel, A, 64)
        self.m3 = Ske_MixF(base_channel, base_channel, A, 64)
        self.m4 = Ske_MixF(base_channel, base_channel, A, 64)
        self.m5 = Ske_MixF(base_channel, base_channel * 2, A, 32, stride=2)
        self.m6 = Ske_MixF(base_channel * 2, base_channel * 2, A, 32)
        self.m7 = Ske_MixF(base_channel * 2, base_channel * 2, A, 32)
        self.m8 = Ske_MixF(base_channel * 2, base_channel * 4, A, 16, stride=2)
        self.m9 = Ske_MixF(base_channel * 4, base_channel * 4, A, 16)
        self.m10 = Ske_MixF(base_channel * 4, base_channel * 4, A, 16)

        self.fc = nn.Linear(base_channel*4, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)

        self.alpha1 = nn.Parameter(torch.ones(1))
        self.alpha2 = nn.Parameter(torch.ones(1))
        self.alpha3 = nn.Parameter(torch.ones(1))
        self.first_tram = nn.Sequential(
            nn.AvgPool2d((4, 1)),
            nn.Conv2d(base_channel, base_channel * 4, 1),
            nn.BatchNorm2d(base_channel * 4),
            nn.ReLU()
        )
        self.second_tram = nn.Sequential(
            nn.AvgPool2d((2, 1)),
            nn.Conv2d(base_channel * 2, base_channel * 4, 1),
            nn.BatchNorm2d(base_channel * 4),
            nn.ReLU()
        )
        if drop_out:
            self.drop_out = nn.Dropout(drop_out)
        else:
            self.drop_out = lambda x: x

    def forward(self, x):
        if len(x.shape) == 3:
            N, T, VC = x.shape
            x = x.view(N, T, self.num_point, -1).permute(0, 3, 1, 2).contiguous().unsqueeze(-1)
        N, C, T, V, M = x.size()

        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)
        x1 = x.clone()
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)
        x = self.l9(x)
        x = self.l10(x)

        x1 = self.m1(x1)
        x1 = self.m2(x1)
        x1 = self.m3(x1)
        x1 = self.m4(x1)
        x2 = x1
        x1 = self.m5(x1)
        x1 = self.m6(x1)
        x1 = self.m7(x1)
        x3 = x1
        x1 = self.m8(x1)
        x1 = self.m9(x1)
        x1 = self.m10(x1)

        x2 = self.first_tram(x2)  # x2(N*M,64,75,25)
        x3 = self.second_tram(x3)  # x3(N*M,128,75,25)
        x = x + self.alpha1 * x1 + self.alpha2 * x2 + self.alpha3 * x3

        # N*M,C,T,V
        c_new = x.size(1)
        x = x.view(N, M, c_new, -1)
        x = x.mean(3).mean(1)
        x = self.drop_out(x)

        return self.fc(x)