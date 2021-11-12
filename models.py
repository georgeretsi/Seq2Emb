import torch.nn as nn
import torch.nn.functional as F
import torch


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class CNN(nn.Module):
    def __init__(self, cnn_cfg):
        super(CNN, self).__init__()

        self.features = nn.ModuleList([])

        self.features.add_module('conv0', nn.Sequential(nn.Conv2d(1, 32, 7, 2, 3), nn.ReLU()))
        in_channels = 32
        cntm = 0
        cnt = 1
        for m in cnn_cfg:
            if m == 'M':
                self.features.add_module('mxp' + str(cntm), nn.MaxPool2d(kernel_size=2, stride=2))
                cntm += 1
            else:
                for i in range(m[0]):
                    x = m[1]
                    self.features.add_module('cnv' + str(cnt), BasicBlock(in_channels, x))
                    in_channels = x
                    cnt += 1

    def forward(self, x, reduce=True):

        y = x
        for nn_module in self.features:
            y = nn_module(y)

        if reduce:
            height = y.size(2)
            y = F.max_pool2d(y, [y.size(2), 3], stride=[y.size(2), 1], padding=[0, 1])
        else:
            y = F.avg_pool2d(y, [7, 1], stride=[1, 1], padding=[3, 0])

        y = y.permute(2, 3, 0, 1)[0]
        return y

def weight_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight.data)


class CTCtop(nn.Module):
    def __init__(self, input_size, hidden_size, nclasses):
        super(CTCtop, self).__init__()

        self.temporal = nn.Sequential(
            nn.Conv2d(input_size, hidden_size, kernel_size=(1,5), stride=(1,1), padding=(0,2)),
            nn.BatchNorm2d(hidden_size), nn.ReLU(), nn.Dropout(.25),
            nn.Conv2d(hidden_size, hidden_size, kernel_size=(1, 5), stride=(1, 1), padding=(0, 2)),
            nn.BatchNorm2d(hidden_size), nn.ReLU(), nn.Dropout(.25),
            nn.Conv2d(hidden_size, nclasses, kernel_size=(1, 5), stride=1, padding=(0, 2)),
        )

    def forward(self, x):

        y = x.unsqueeze(0).permute(2, 3, 0, 1)
        y = self.temporal(y).permute(2, 3, 0, 1)
        return y


class Enc(nn.Module):
    def __init__(self, rnn_cfg, nin, nout, nclasses=None):
        super(Enc, self).__init__()

        hidden, num_layers = rnn_cfg
        self.enc = nn.GRU(nin, hidden, num_layers=num_layers, bidirectional=True, dropout=.1)
        self.rnn_out = 2 * hidden * num_layers

        self.fnl = nn.Sequential(nn.Dropout(.1), nn.Linear(self.rnn_out, nout))#, nn.BatchNorm1d(nout))

        if nclasses is None:
            self.ctc = None
        else:
            self.ctc = nn.Sequential(nn.Dropout(.1), nn.Linear(2 * hidden, nclasses))


    def forward(self, x):
        # encode predictions:
        o, y = self.enc(x) # hidden encoding
        y = y.permute(1, 0, 2).contiguous().view(-1, self.rnn_out)
        y = self.fnl(y)

        if self.ctc is not None:
            o = self.ctc(o)
            return o, y
        else:
            return y

class DecoderChar(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderChar, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, dropout=.0)
        self.out = nn.Sequential(nn.Dropout(.1), nn.Linear(hidden_size, output_size))
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):

        output = self.embedding(input).view(1, input.size(0), -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self, device):
        return torch.zeros(1, 1, self.hidden_size).to(device)


class EncoderChar(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(EncoderChar, self).__init__()
        self.hidden_size = hidden_size
        num_layers = 2

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers=num_layers, bidirectional=True, dropout=.1)
        self.rnn_out = 2 * hidden_size * num_layers
        self.fnl = nn.Sequential(#nn.Dropout(.1), nn.Linear(self.rnn_out, output_size), nn.ReLU(),
                                 #nn.Dropout(.1), nn.Linear(output_size, output_size),
                                 nn.Dropout(.1), nn.Linear(self.rnn_out, output_size),
                                 )

    def forward(self, input):
        output = self.embedding(input).permute(1, 0, 2)
        output = F.relu(output)
        _, hidden = self.gru(output)
        hidden = hidden.permute(1, 0, 2).contiguous().view(-1, self.rnn_out)
        output = self.fnl(hidden)
        return output


class SignSmooth(nn.Module):
    def __init__(self, a=1):
        super(SignSmooth, self).__init__()

        self.a = a

    def forward(self, x):
        y = Sign.apply(F.tanh(self.a * x))
        return y

class Sign(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        return x.sign()

    @staticmethod
    def backward(ctx, g):
        return g
