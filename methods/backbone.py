import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.nn.utils import weight_norm

# --- gaussian initialize ---
def init_layer(L):
  if isinstance(L, nn.Conv2d):
    n = L.kernel_size[0]*L.kernel_size[1]*L.out_channels
    L.weight.data.normal_(0,math.sqrt(2.0/float(n)))
  elif isinstance(L, nn.BatchNorm2d):
    L.weight.data.fill_(1)
    L.bias.data.fill_(0)

class distLinear(nn.Module):
  def __init__(self, indim, outdim):
    super(distLinear, self).__init__()
    self.L = weight_norm(nn.Linear(indim, outdim, bias=False), name='weight', dim=0)
    self.relu = nn.ReLU()

  def forward(self, x):
    x_norm = torch.norm(x, p=2, dim =1).unsqueeze(1).expand_as(x)
    x_normalized = x.div(x_norm + 0.00001)
    L_norm = torch.norm(self.L.weight.data, p=2, dim =1).unsqueeze(1).expand_as(self.L.weight.data)
    self.L.weight.data = self.L.weight.data.div(L_norm + 0.00001)
    cos_dist = self.L(x_normalized)
    scores = 10 * cos_dist
    return scores

# --- flatten tensor ---
class Flatten(nn.Module):
  def __init__(self):
    super(Flatten, self).__init__()

  def forward(self, x):
    return x.view(x.size(0), -1)

# --- LSTMCell module for matchingnet ---
class LSTMCell(nn.Module):
  FWT = False
  def __init__(self, input_size, hidden_size, bias=True):
    super(LSTMCell, self).__init__()
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.bias = bias
    if self.FWT:
      self.x2h = Linear_fw(input_size, 4 * hidden_size, bias=bias)
      self.h2h = Linear_fw(hidden_size, 4 * hidden_size, bias=bias)
    else:
      self.x2h = nn.Linear(input_size, 4 * hidden_size, bias=bias)
      self.h2h = nn.Linear(hidden_size, 4 * hidden_size, bias=bias)
    self.reset_parameters()

  def reset_parameters(self):
    std = 1.0 / math.sqrt(self.hidden_size)
    for w in self.parameters():
      w.data.uniform_(-std, std)

  def forward(self, x, hidden=None):
    if hidden is None:
      hx = torch.zeors_like(x)
      cx = torch.zeros_like(x)
    else:
      hx, cx = hidden

    gates = self.x2h(x) + self.h2h(hx)
    ingate, forgetgate, cellgate, outgate = torch.split(gates, self.hidden_size, dim=1)

    ingate = torch.sigmoid(ingate)
    forgetgate = torch.sigmoid(forgetgate)
    cellgate = torch.tanh(cellgate)
    outgate = torch.sigmoid(outgate)

    cy = torch.mul(cx, forgetgate) +  torch.mul(ingate, cellgate)
    hy = torch.mul(outgate, torch.tanh(cy))
    return (hy, cy)

# --- LSTM module for matchingnet ---
class LSTM(nn.Module):
  def __init__(self, input_size, hidden_size, num_layers=1, bias=True, batch_first=False, bidirectional=False):
    super(LSTM, self).__init__()
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.bias = bias
    self.batch_first = batch_first
    self.num_directions = 2 if bidirectional else 1
    assert(self.num_layers == 1)

    self.lstm = LSTMCell(input_size, hidden_size, self.bias)

  def forward(self, x, hidden=None):
    # swap axis if batch first
    if self.batch_first:
      x = x.permute(1, 0 ,2)

    # hidden state
    if hidden is None:
      h0 = torch.zeros(self.num_directions, x.size(1), self.hidden_size, dtype=x.dtype, device=x.device)
      c0 = torch.zeros(self.num_directions, x.size(1), self.hidden_size, dtype=x.dtype, device=x.device)
    else:
      h0, c0 = hidden

    # forward
    outs = []
    hn = h0[0]
    cn = c0[0]
    for seq in range(x.size(0)):
      hn, cn = self.lstm(x[seq], (hn, cn))
      outs.append(hn.unsqueeze(0))
    outs = torch.cat(outs, dim=0)

    # reverse foward
    if self.num_directions == 2:
      outs_reverse = []
      hn = h0[1]
      cn = c0[1]
      for seq in range(x.size(0)):
        seq = x.size(1) - 1 - seq
        hn, cn = self.lstm(x[seq], (hn, cn))
        outs_reverse.append(hn.unsqueeze(0))
      outs_reverse = torch.cat(outs_reverse, dim=0)
      outs = torch.cat([outs, outs_reverse], dim=2)

    # swap axis if batch first
    if self.batch_first:
      outs = outs.permute(1, 0, 2)
    return outs

# --- Linear module ---
class Linear_fw(nn.Linear):
  def __init__(self, in_features, out_features, bias=True):
    super(Linear_fw, self).__init__(in_features, out_features, bias=bias)
    self.weight.fast = None
    self.bias.fast = None

  def forward(self, x):
    if self.weight.fast is not None and self.bias.fast is not None:
      out = F.linear(x, self.weight.fast, self.bias.fast)
    else:
      out = super(Linear_fw, self).forward(x)
    return out

# --- Conv2d module ---
class Conv2d_fw(nn.Conv2d):
  def __init__(self, in_channels, out_channels, kernel_size, stride=1,padding=0, bias = True):
    super(Conv2d_fw, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
    self.weight.fast = None
    if not self.bias is None:
      self.bias.fast = None

  def forward(self, x):
    if self.bias is None:
      if self.weight.fast is not None:
        out = F.conv2d(x, self.weight.fast, None, stride= self.stride, padding=self.padding)
      else:
        out = super(Conv2d_fw, self).forward(x)
    else:
      if self.weight.fast is not None and self.bias.fast is not None:
        out = F.conv2d(x, self.weight.fast, self.bias.fast, stride= self.stride, padding=self.padding)
      else:
        out = super(Conv2d_fw, self).forward(x)
    return out

# --- softplus module ---
def softplus(x):
  return torch.nn.functional.softplus(x, beta=100)

# --- feature-wise transformation layer ---
class FeatureWiseTransformation2d_fw(nn.BatchNorm2d):
  feature_augment = False
  GRAM = False
  NON = False
  def __init__(self, num_features, momentum=0.1, track_running_stats=True):
    super(FeatureWiseTransformation2d_fw, self).__init__(num_features, momentum=momentum, track_running_stats=track_running_stats)
    self.weight.fast = None
    self.bias.fast = None
    if self.track_running_stats:
      self.register_buffer('running_mean', torch.zeros(num_features))
      self.register_buffer('running_var', torch.zeros(num_features))
    if self.feature_augment:  # initialize {gamma, beta} with {0.3, 0.5}
      self.gamma = torch.nn.Parameter(torch.ones(1, num_features, 1, 1)*0.3)
      self.beta  = torch.nn.Parameter(torch.ones(1, num_features, 1, 1)*0.5)
    if self.GRAM:
      self.gamma = torch.nn.Parameter(torch.ones(1, num_features, 1, 1)*0.3)
      self.beta  = torch.nn.Parameter(torch.ones(1, num_features, 1, 1)*0.5)
    if self.NON:
      self.Conv = Conv2d_fw(num_features, num_features, kernel_size=3, padding=1, bias=True)
    self.grl = GRL()
    self.reset_parameters()

  def reset_running_stats(self):
    if self.track_running_stats:
      self.running_mean.zero_()
      self.running_var.fill_(1)

  def forward(self, x, epoch=0):
    if self.weight.fast is not None and self.bias.fast is not None:
      weight = self.weight.fast
      bias = self.bias.fast
    else:
      weight = self.weight
      bias = self.bias
    if self.track_running_stats:
      out = F.batch_norm(x, self.running_mean, self.running_var, weight, bias, training=self.training, momentum=self.momentum)
    else:
      out = F.batch_norm(x, torch.zeros_like(x), torch.ones_like(x), weight, bias, training=True, momentum=1)

    # apply feature-wise transformation
    if self.feature_augment and self.training:
      eta = -1
      lamb = 2/(1 + math.exp(eta * (epoch / 400) ) ) - 1
      if self.NON == False:
        gamma = (1 + torch.randn(1, self.num_features, 1, 1, dtype=self.gamma.dtype, device=self.gamma.device)*softplus(self.gamma)).expand_as(out)
        beta = (torch.randn(1, self.num_features, 1, 1, dtype=self.beta.dtype, device=self.beta.device)*softplus(self.beta)).expand_as(out)
      # if epoch < 150:
      #   out = gamma * out + beta
      else:
        # self.grl.set_lambda(lamb)
        # out = self.grl(out)
        out = gamma*out + beta
        # self.grl.set_lambda(1.0/lamb)
        # out = self.grl(out)
      # self.grl.set_lambda(1.0)
      # out = self.grl(out)
      # out = gamma*out + beta
      # self.grl.set_lambda(1.0)
      # out = self.grl(out)
      return out
    if self.GRAM and self.training:
      gamma = (1 + torch.randn(1, self.num_features, 1, 1, dtype=self.gamma.dtype, device=self.gamma.device)*softplus(self.gamma)).expand_as(out)
      beta = (torch.randn(1, self.num_features, 1, 1, dtype=self.beta.dtype, device=self.beta.device)*softplus(self.beta)).expand_as(out)
      out_ND = out
      eta = -1
      lamb = 2/(1 + math.exp(eta * (epoch / 400) ) ) - 1
      if self.NON == True:
        self.grl.set_lambda(1.0)
        out_ND = self.grl(out_ND)
        out_ND == self.Conv(out_ND)
        self.grl.set_lambda(1.0)
        out_ND = self.grl(out_ND)
      elif epoch < 250:
        # self.grl.set_lambda(100)
        # out_ND = self.grl(out_ND)
        # if self.NON==False:
        out_ND = gamma*out_ND + beta
        # else:
        #   out_ND = self.Conv(out_ND)
        # self.grl.set_lambda(0.01)
        # out_ND = self.grl(out_ND)
      else:
        self.grl.set_lambda(1.0)
        out_ND = self.grl(out_ND)
        # if self.NON == True:
        #   out_ND = self.Conv(out_ND)
        # else:
        out_ND = gamma*out_ND + beta
        # out_ND = self.Conv(out_ND)
        # out_ND = self.Conv2(out_ND)
        self.grl.set_lambda(1.0)
        out_ND = self.grl(out_ND)
      return out_ND, out
    return out

# --- BatchNorm2d ---
class BatchNorm2d_fw(nn.BatchNorm2d):
  def __init__(self, num_features, momentum=0.1, track_running_stats=True):
    super(BatchNorm2d_fw, self).__init__(num_features, momentum=momentum, track_running_stats=track_running_stats)
    self.weight.fast = None
    self.bias.fast = None
    if self.track_running_stats:
      self.register_buffer('running_mean', torch.zeros(num_features))
      self.register_buffer('running_var', torch.zeros(num_features))
    self.reset_parameters()

  def reset_running_stats(self):
    if self.track_running_stats:
      self.running_mean.zero_()
      self.running_var.fill_(1)

  def forward(self, x, step=0):
    if self.weight.fast is not None and self.bias.fast is not None:
      weight = self.weight.fast
      bias = self.bias.fast
    else:
      weight = self.weight
      bias = self.bias
    if self.track_running_stats:
      out = F.batch_norm(x, self.running_mean, self.running_var, weight, bias, training=self.training, momentum=self.momentum)
    else:
      out = F.batch_norm(x, torch.zeros(x.size(1), dtype=x.dtype, device=x.device), torch.ones(x.size(1), dtype=x.dtype, device=x.device), weight, bias, training=True, momentum=1)
    return out

# --- BatchNorm1d ---
class BatchNorm1d_fw(nn.BatchNorm1d):
  def __init__(self, num_features, momentum=0.1, track_running_stats=True):
    super(BatchNorm1d_fw, self).__init__(num_features, momentum=momentum, track_running_stats=track_running_stats)
    self.weight.fast = None
    self.bias.fast = None
    if self.track_running_stats:
      self.register_buffer('running_mean', torch.zeros(num_features))
      self.register_buffer('running_var', torch.zeros(num_features))
    self.reset_parameters()

  def reset_running_stats(self):
    if self.track_running_stats:
      self.running_mean.zero_()
      self.running_var.fill_(1)

  def forward(self, x, step=0):
    if self.weight.fast is not None and self.bias.fast is not None:
      weight = self.weight.fast
      bias = self.bias.fast
    else:
      weight = self.weight
      bias = self.bias
    if self.track_running_stats:
      out = F.batch_norm(x, self.running_mean, self.running_var, weight, bias, training=self.training, momentum=self.momentum)
    else:
      out = F.batch_norm(x, torch.zeros(x.size(1), dtype=x.dtype, device=x.device), torch.ones(x.size(1), dtype=x.dtype, device=x.device), weight, bias, training=True, momentum=1)
    return out

#--- Add Layer replace the + in the residule block----
class Add(nn.Module):
    def __init__(self):
        super(Add, self).__init__()

    def forward(self, x, y):
        return x+y

# --- Simple Conv Block ---
class ConvBlock(nn.Module):
  FWT = False
  GRAM = False
  def __init__(self, indim, outdim, pool = True, padding = 1):
    super(ConvBlock, self).__init__()
    self.indim  = indim
    self.outdim = outdim
    if self.FWT:
      self.C = Conv2d_fw(indim, outdim, 3, padding = padding)
      self.BN = FeatureWiseTransformation2d_fw(outdim)
    elif self.GRAM:
      self.C = nn.Conv2d(indim, outdim, 3, padding= padding)
      self.BN = nn.BatchNorm2d(outdim)
      self.C_Gram = Conv2d_fw(indim, outdim, 3, padding = padding)
      self.BN_Gram = FeatureWiseTransformation2d_fw(outdim)
    else:
      self.C = nn.Conv2d(indim, outdim, 3, padding= padding)
      self.BN = nn.BatchNorm2d(outdim)
    self.relu = nn.ReLU(inplace=True)

    self.parametrized_layers = [self.C, self.BN, self.relu]
    if self.GRAM:
      self.parametrized_layers_Gram = [self.C_Gram, self.BN_Gram, self.relu]
    if pool:
      self.pool = nn.MaxPool2d(2)
      self.parametrized_layers.append(self.pool)
      if self.GRAM:
        self.parametrized_layers_Gram.append(self.pool)

    for layer in self.parametrized_layers:
      init_layer(layer)
    if self.GRAM:
      for layer in self.parametrized_layers:
        init_layer(layer)
    self.trunk = nn.Sequential(*self.parametrized_layers)
    if self.GRAM:
      self.trunk_Gram = nn.Sequential(*self.parametrized_layers_Gram)

  def forward(self,x):
    out = self.trunk(x)
    if self.GRAM:
      out_D = trunk_Gram(x)
      return out, out_D
    return out

# --- Simple ResNet Block ---
class SimpleBlock(nn.Module):
  FWT = False
  GRAM = False
  def __init__(self, indim, outdim, half_res, leaky=False):
    super(SimpleBlock, self).__init__()
    self.indim = indim
    self.outdim = outdim
    if self.FWT:
      self.C1 = Conv2d_fw(indim, outdim, kernel_size=3, stride=2 if half_res else 1, padding=1, bias=False)
      self.BN1 = BatchNorm2d_fw(outdim)
      self.C2 = Conv2d_fw(outdim, outdim,kernel_size=3, padding=1,bias=False)
      self.BN2 = FeatureWiseTransformation2d_fw(outdim) # feature-wise transformation at the end of each residual block
    elif self.GRAM:
      self.C1 = Conv2d_fw(indim, outdim, kernel_size=3, stride=2 if half_res else 1, padding=1, bias=False)
      self.BN1 = BatchNorm2d_fw(outdim)
      self.C2 = Conv2d_fw(outdim, outdim,kernel_size=3, padding=1,bias=False)
      self.BN2 = FeatureWiseTransformation2d_fw(outdim) # feature-wise transformation at the end of each residual block

      # self.C1 = nn.Conv2d(indim, outdim, kernel_size=3, stride=2 if half_res else 1, padding=1, bias=False)
      # self.BN1 = nn.BatchNorm2d(outdim)
      # self.C2 = nn.Conv2d(outdim, outdim,kernel_size=3, padding=1,bias=False)
      # self.BN2 = nn.BatchNorm2d(outdim)

    else:
      self.C1 = nn.Conv2d(indim, outdim, kernel_size=3, stride=2 if half_res else 1, padding=1, bias=False)
      self.BN1 = nn.BatchNorm2d(outdim)
      self.C2 = nn.Conv2d(outdim, outdim,kernel_size=3, padding=1,bias=False)
      self.BN2 = nn.BatchNorm2d(outdim)

      # self.C1_ND = nn.Conv2d(indim, outdim, kernel_size=3, stride=2 if half_res else 1, padding=1, bias=False)
      # self.BN1_ND = nn.BatchNorm2d(outdim)
      # self.C2_ND = nn.Conv2d(outdim, outdim,kernel_size=3, padding=1,bias=False)
      # self.BN2_ND = nn.BatchNorm2d(outdim) # feature-wise transformation at the end of each residual block

    self.relu1 = nn.ReLU(inplace=True) if not leaky else nn.LeakyReLU(0.2, inplace=True)
    self.relu2 = nn.ReLU(inplace=True) if not leaky else nn.LeakyReLU(0.2, inplace=True)
    self.grl = GRL()

    self.parametrized_layers = [self.C1, self.C2, self.BN1, self.BN2]
    
    self.half_res = half_res

    # if the input number of channels is not equal to the output, then need a 1x1 convolution
    if indim!=outdim:
      if self.FWT:
        self.shortcut = Conv2d_fw(indim, outdim, 1, 2 if half_res else 1, bias=False)
        self.BNshortcut = FeatureWiseTransformation2d_fw(outdim)
      elif self.GRAM:
        self.shortcut = Conv2d_fw(indim, outdim, 1, 2 if half_res else 1, bias=False)
        self.BNshortcut = FeatureWiseTransformation2d_fw(outdim)
        # self.shortcut = nn.Conv2d(indim, outdim, 1, 2 if half_res else 1, bias=False)
        # self.BNshortcut = nn.BatchNorm2d(outdim)
      else:
        # self.shortcut_ND = nn.Conv2d(indim, outdim, 1, 2 if half_res else 1, bias=False)
        # self.BNshortcut_ND = nn.BatchNorm2d(outdim)
        self.shortcut = nn.Conv2d(indim, outdim, 1, 2 if half_res else 1, bias=False)
        self.BNshortcut = nn.BatchNorm2d(outdim)

      self.parametrized_layers.append(self.shortcut)
      self.parametrized_layers.append(self.BNshortcut)
      # if self.GRAM:
      #   self.parametrized_layers.append(self.shortcut_ND)
      #   self.parametrized_layers.append(self.BNshortcut_ND)
      self.shortcut_type = '1x1'
    else:
      self.shortcut_type = 'identity'

    for layer in self.parametrized_layers:
      init_layer(layer)
    # if self.GRAM:
    #   for layer in self.parametrized_layers_ND:
    #     init_layer(layer)

  def forward(self, x, y = None, epoch = 0):
    if y == None:
      out = self.C1(x)
      out = self.BN1(out)
      out = self.relu1(out)
      out = self.C2(out)
      if epoch > 0:
        out = self.BN2(out, epoch = epoch)
      else:
        out = self.BN2(out)
      short_out = x if self.shortcut_type == 'identity' else self.BNshortcut(self.shortcut(x))
      out = out + short_out
      out = self.relu2(out)
      return out
      
    else:
      # print('yes')
      out_ND = self.C1(x)
      out = self.C1(y)
      out_ND = self.BN1(out_ND)
      out = self.BN1(out)
      out_ND = self.relu1(out_ND)
      out = self.relu1(out)
      out_ND = self.C2(out_ND)
      out = self.C2(out)
      out_ND, _ = self.BN2(out_ND, epoch)
      _, out = self.BN2(out, epoch)
      if self.shortcut_type == 'identity':
        short_out_ND = x
        short_out = y
      else:
        short_out_ND, _ = self.BNshortcut(self.shortcut(x), epoch)
        _, short_out = self.BNshortcut(self.shortcut(y), epoch)
      # short_out = x if self.shortcut_type == 'identity' else self.BNshortcut(self.shortcut(x))
      out = out + short_out
      out_ND = out_ND + short_out_ND
      out = self.relu2(out)
      out_ND = self.relu2(out_ND)
      return out_ND, out
      # if epoch % 2 == 0:
      #   out_ND = self.C1(x)
      #   out = self.C1(y)
      #   out_ND = self.BN1(out_ND)
      #   out = self.BN1(out)
      #   out_ND = self.relu1(out_ND)
      #   out = self.relu1(out)
      #   out_ND = self.C2(out_ND)
      #   out = self.C2(out)
      #   out_ND, _ = self.BN2(out_ND, epoch)
      #   _, out = self.BN2(out, epoch)
      #   if self.shortcut_type == 'identity':
      #     short_out_ND = x
      #     short_out = y
      #   else:
      #     short_out_ND, _ = self.BNshortcut(self.shortcut(x), epoch)
      #     _, short_out = self.BNshortcut(self.shortcut(y), epoch)
      #   # short_out = x if self.shortcut_type == 'identity' else self.BNshortcut(self.shortcut(x))
      #   out = out + short_out
      #   out_ND = out_ND + short_out_ND
      #   out = self.relu2(out)
      #   out_ND = self.relu2(out_ND)
      #   return out_ND, out
      # else:
      #   out = self.C1(x)
      #   out_ND = self.C1(y)
      #   out = self.BN1(out)
      #   out_ND = self.BN1(out_ND)
      #   out = self.relu1(out)
      #   out_ND = self.relu1(out_ND)
      #   out_ND = self.C2(out_ND)
      #   out = self.C2(out)
      #   out_ND, _ = self.BN2(out_ND, epoch)
      #   _, out = self.BN2(out, epoch)
      #   if self.shortcut_type == 'identity':
      #     short_out = x
      #     short_out_ND = y
      #   else:
      #     short_out_ND, _ = self.BNshortcut(self.shortcut(y), epoch)
      #     _, short_out = self.BNshortcut(self.shortcut(x), epoch)
      #   # short_out = x if self.shortcut_type == 'identity' else self.BNshortcut(self.shortcut(x))
      #   out = out + short_out
      #   out_ND = out_ND + short_out_ND
      #   out = self.relu2(out)
      #   out_ND = self.relu2(out_ND)
      #   return out, out_ND
    # out_ND = self.C1_ND(x)
    # out_ND = self.BN1_ND(out_ND)
    # out_ND = self.relu1(out_ND)
    # out_ND = self.C2_ND(out_ND)
    # out_ND = self.BN2_ND(out_ND)
    # short_out = x if self.shortcut_type == 'identity' else self.BNshortcut_ND(self.shortcut_ND(x))
    # out_ND = out + short_out
    # out_ND = self.relu2(out_ND)
    # out = out + short_out
    # out = self.relu2(out)

    # if y == None:
    #   # out = (out + out_ND)/2
    #   return out
    # else:
    #   out_ND = self.C1_ND(y)
    #   self.grl.set_lambda(1.0)
    #   out_ND = self.grl(out_ND)
    #   out_ND = self.BN1_ND(out_ND)
    #   self.grl.set_lambda(1.0)
    #   out_ND = self.grl(out_ND)
    #   out_ND = self.relu1(out_ND)
    #   out_ND = self.C2_ND(out_ND)
    #   self.grl.set_lambda(1.0)
    #   out_ND = self.grl(out_ND)
    #   out_ND = self.BN2_ND(out_ND)
    #   self.grl.set_lambda(1.0)
    #   out_ND = self.grl(out_ND)
    #   short_out = x if self.shortcut_type == 'identity' else self.BNshortcut_ND(self.shortcut_ND(x))
    #   out_ND = out_ND + short_out
    #   out_ND = self.relu2(out_ND)

    #   return out, out_ND

# --- ConvNet module ---
class ConvNet(nn.Module):
  def __init__(self, depth, flatten = True):
    super(ConvNet,self).__init__()
    self.grads = []
    self.fmaps = []
    trunk = []
    for i in range(depth):
      indim = 3 if i == 0 else 64
      outdim = 64
      B = ConvBlock(indim, outdim, pool = ( i <4 ) ) #only pooling for fist 4 layers
      trunk.append(B)

    if flatten:
      trunk.append(Flatten())

    self.trunk = nn.Sequential(*trunk)
    self.final_feat_dim = 1600

  def forward(self,x):
    out = self.trunk(x)
    return out

# --- ConvNetNopool module ---
class ConvNetNopool(nn.Module): #Relation net use a 4 layer conv with pooling in only first two layers, else no pooling
  def __init__(self, depth):
    super(ConvNetNopool,self).__init__()
    self.grads = []
    self.fmaps = []
    trunk = []
    for i in range(depth):
      indim = 3 if i == 0 else 64
      outdim = 64
      B = ConvBlock(indim, outdim, pool = ( i in [0,1] ), padding = 0 if i in[0,1] else 1  ) #only first two layer has pooling and no padding
      trunk.append(B)

    self.trunk = nn.Sequential(*trunk)
    self.final_feat_dim = [64,19,19]

  def forward(self,x):
    out = self.trunk(x)
    return out

# --- ResNet module ---
class ResNet(nn.Module):
  FWT = False
  GRAM = False
  def __init__(self,block,list_of_num_layers, list_of_out_dims, flatten=True, leakyrelu=False):
    # list_of_num_layers specifies number of layers in each stage
    # list_of_out_dims specifies number of output channel for each stage
    super(ResNet,self).__init__()
    self.grads = []
    self.fmaps = []
    assert len(list_of_num_layers)==4, 'Can have only four stages'
    if self.FWT or self.GRAM:
      conv1 = Conv2d_fw(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
      bn1 = BatchNorm2d_fw(64)
    # elif self.GRAM:
    #   conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
    #   bn1 = nn.BatchNorm2d(64)
    #   conv1_ND = Conv2d_fw(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
    #   bn1_ND = BatchNorm2d_fw(64)
    else:
      conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
      bn1 = nn.BatchNorm2d(64)

    relu = nn.ReLU(inplace=True) if not leakyrelu else nn.LeakyReLU(0.2, inplace=True)
    pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    init_layer(conv1)
    init_layer(bn1)
    # init_layer(conv1_ND)
    # init_layer(bn1_ND)

    trunk = [conv1, bn1, relu, pool1]
    # if self.GRAM:
    #   trunk_ND = [conv1_ND, bn1_ND, relu, pool1]

    indim = 64
    for i in range(4):
      for j in range(list_of_num_layers[i]):
        half_res = (i>=1) and (j==0)
        B = block(indim, list_of_out_dims[i], half_res, leaky=leakyrelu)
        trunk.append(B)
        indim = list_of_out_dims[i]

    self.avgpool = nn.AvgPool2d(7)
    self.Flatten = Flatten()
    self.Flat = flatten
    if flatten:
    #   avgpool = nn.AvgPool2d(7)
    #   trunk.append(avgpool)
    #   trunk.append(Flatten())
      self.final_feat_dim = indim
    else:
      self.final_feat_dim = [ indim, 7, 7]
    # self.final_feat_dim = indim
    
    self.Domain_cls = Domain_Cls()
      # self.Domain_cls2 = Domain_cls()

      #channel size is 512
    self.CAtt = ChannelAttention(512)
    self.DAtt = ChannelAttention(512)
    self.MAtt = MixChannelAttention(512)

    self.ND = Gram()

    self.trunk = nn.Sequential(*trunk)
    

  def forward(self, x, ND = False, epoch = 0, NON = False):
    
    if ND == False:
      # print(self.trunk)
      out_BF = x
      for i in range(len(self.trunk)):
        if i < 4:
          out_BF = self.trunk[i](out_BF)
          continue
        out_BF = self.trunk[i](out_BF, epoch = epoch)
      # out_BF = self.trunk(x)
      out = out_BF
      if self.Flat:
        out = self.avgpool(out)
        out = self.Flatten(out)
      
      return out
    else:
      Gram_loss = 0
      out_ND = x
      out = x
      for i in range(len(self.trunk)):
        if i == 4:
          out_ND, out = self.trunk[i](out_ND,out_ND, epoch = epoch)
          out_4 = out_ND # batch_size * 64 * 56 * 56
          Gram_loss += self.ND(out, out_ND, 4)
        elif i == 5:
          out_ND, out = self.trunk[i](out_ND, out, epoch = epoch)
          out_5 = out_ND # batch_size * 128 * 28 * 28
          Gram_loss += self.ND(out, out_ND, 5)
        elif i == 6:
          out_ND, out = self.trunk[i](out_ND, out, epoch = epoch)
          out_6 = out_ND # batch_size * 256 * 14 * 14
          Gram_loss += self.ND(out, out_ND, 6) 
        elif i == 7:
          out_ND, out = self.trunk[i](out_ND,out, epoch = epoch)
          out_7 = out_ND # batch_size * 512 * 7 * 7
          Gram_loss += self.ND(out, out_ND, 7)
        else:
          out_ND = self.trunk[i](out_ND)
          out = self.trunk[i](out)

      Gram_loss /= 4
      out_BF = out_ND
      out_BF_O = out
      # if self.Flat:
      if out == None:
        out = out_ND
      out = self.avgpool(out)
      out = self.Flatten(out)
      out_ND = self.avgpool(out_ND)
      out_ND = self.Flatten(out_ND)

      DC_loss = 0
      out_D = torch.cat([out,out_ND],dim = 0)
      nums = out.shape[0]
      domain_label = torch.zeros(2*nums)
      domain_label[nums:] = 1
      domain_label = domain_label.cuda()
      Score_D , DC_loss = self.Domain_cls(out_D,domain_label, epoch = epoch)

      eta = -1
      lamb = 2/(1 + math.exp(eta * (epoch / 400) ) ) - 1
      DC_loss += (lamb * Gram_loss)
      if self.Flat:
        return out_ND, DC_loss
        # return out, DC_loss
      else:
        return out_BF, DC_loss
        # return out_BF_O, DC_loss

    # out = self.trunk(x)
    # return out


class Domain_Cls(nn.Module):
  def __init__(self, GradRL = True):
    super(Domain_Cls,self).__init__()
    # self.soft = nn.Sigmoid()
    # self.d_cls = nn.Sequential()
    # self.d_cls.add_module('fc',nn.Linear(512,100))
    # self.d_cls.add_module('batchn',nn.BatchNorm1d(100))
    # self.d_cls.add_module('d_relu',nn.ReLU(True))
    # self.d_cls.add_module('d_fc',nn.Linear(100,2)) 
    # self.d_cls.add_module('activate',nn.Softmax(dim=1))

    self.fc = nn.Linear(512,100)
    self.batchn = nn.BatchNorm1d(100)
    self.d_relu = nn.ReLU(True)
    # self.miu = nn.Parameter(torch.ones(1, 512)*0.3)
    # self.theta  = nn.Parameter(torch.ones(1, 512)*0.5)
    self.fc2 = nn.Linear(100,2)
    self.activate = nn.Softmax(dim=1)
    self.loss = nn.CrossEntropyLoss()
    self.GradRL = GradRL
    if self.GradRL:
      self.grl = GRL()

  def forward(self,x,domain_label, epoch = 0):
    eta = -1
    lamb = 2/(1 + math.exp(eta * (epoch / 400) ) ) - 1
    if self.GradRL:
      self.grl.set_lambda(lamb)
      x = self.grl(x)
    # self.n = x.shape[0]
    # out = self.d_cls(x)
    out = self.fc(x)
    out = self.batchn(out)
    out = self.d_relu(out)
    out = self.fc2(out)
    out = self.activate(out)
    # out = self.activate(self.fc((miu*x + theta).view(-1,512)))
    if domain_label == None:
      return out
    else:
      DC_loss = self.loss(out, domain_label.long())

      return out, DC_loss
      


class grl_func(torch.autograd.Function):
    def __init__(self):
        super(grl_func, self).__init__()

    @ staticmethod
    def forward(ctx, x, lambda_):
        ctx.save_for_backward(lambda_)
        return x.view_as(x)

    @ staticmethod
    def backward(ctx, grad_output):
        lambda_, = ctx.saved_variables
        grad_input = grad_output.clone()
        return - lambda_ * grad_input, None


class GRL(nn.Module):
    def __init__(self, lambda_=0.):
        super(GRL, self).__init__()
        self.lambda_ = torch.tensor(lambda_)

    def set_lambda(self, lambda_):
        self.lambda_ = torch.tensor(lambda_)

    def forward(self, x):
        return grl_func.apply(x, self.lambda_)


class ChannelAttention(nn.Module):
  def __init__(self,channel,reduction=16, leakyrelu=False):
    super(ChannelAttention,self).__init__()
    # self.avg_pool = nn.AdaptiveAvgPool2d(1)
    self.fc1 = nn.Linear(channel, channel // reduction)
    self.relu = nn.ReLU(inplace=True) if not leakyrelu else nn.LeakyReLU(0.2, inplace=True)
    self.fc2 = nn.Linear(channel // reduction, channel)
    # self.sig = nn.Sigmoid()
    # self.fc = nn.Sequential(
    #   nn.Linear(channel, channel // reduction, bias = False),
    #   nn.ReLU(inplace=True),
    #   nn.Linear(channel // reduction, channel, bias = False),
    #   nn.Sigmoid()
    # )
  def forward(self,x):
    # print(x.shape)
    b = x.shape[0]
    c = x.shape[1]
    # y = self.avg_pool(x).view(b,c)
    y = self.fc1(x)
    y = self.relu(y)
    y = self.fc2(y)
    y = torch.sigmoid(y).view(b,c)
    return x*y.expand_as(x)

class MixChannelAttention(nn.Module):
  def __init__(self,channel,reduction=16, leakyrelu=False):
    super(MixChannelAttention, self).__init__()
    # self.avg_pool_C = nn.AdaptiveAvgPool2d(1)
    # self.avg_pool_D = nn.AdaptiveAvgPool2d(1)
    self.fc1 = nn.Linear(channel, channel // reduction, bias = False)
    self.relu1 = nn.ReLU(inplace=True) if not leakyrelu else nn.LeakyReLU(0.2, inplace=True)
    self.fc2 = nn.Linear(channel // reduction, channel, bias = False)
    # self.sig1 = nn.Sigmoid()
    self.fc3 = nn.Linear(channel, channel // reduction, bias = False)
    self.relu2 = nn.ReLU(inplace=True) if not leakyrelu else nn.LeakyReLU(0.2, inplace=True)
    self.fc4 = nn.Linear(channel // reduction, channel, bias = False)
    # self.sig2 = nn.Sigmoid()
  def forward(self,x):
    b = x.shape[0]
    c = x.shape[1]
    y_C = self.fc1(x)
    y_C = self.relu1(y_C)
    y_C = self.fc2(y_C)
    y_C = torch.sigmoid(y_C).view(b,c)
    y_D = self.fc3(x)
    y_D = self.relu2(y_D)
    y_D = self.fc4(y_D)
    y_D = torch.sigmoid(x).view(b,c)
    return x*y_C.expand_as(x),x*y_D.expand_as(x),y_C,y_D

def MixChannelAttentionLoss(x, y):
  b = x.shape[0]
  c = x.shape[1]
  y_t = y.view(c,b)
  AttLoss = torch.sum(torch.mm(x, y_t))
  return AttLoss

class Gram(nn.Module):
  def __init__(self):
    super(Gram, self).__init__()
    self.avg_4 = nn.AvgPool2d(56)
    self.avg_5 = nn.AvgPool2d(28)
    self.avg_6 = nn.AvgPool2d(14)
    self.avg_7 = nn.AvgPool2d(7)
    self.Flat = Flatten()
    self.loss_fn = nn.MSELoss()
  def forward(self, feature1, feature2, num):
    
    # if num == 4:
    #   feature1 = self.avg_4(feature1)
    #   feature2 = self.avg_4(feature2)
    # elif num == 5:
    #   feature1 = self.avg_5(feature1)
    #   feature2 = self.avg_5(feature2)
    # elif num == 6:
    #   feature1 = self.avg_6(feature1)
    #   feature2 = self.avg_6(feature2)
    # else:
    #   feature1 = self.avg_7(feature1)
    #   feature2 = self.avg_7(feature2)
    batch_size = feature1.shape[0]
    channel_size = feature1.shape[1]
    feature1 = feature1.view(batch_size, channel_size, -1)
    feature2 = feature2.view(batch_size, channel_size, -1) # batch size * channel size * (H * W)
    size = feature2.shape[2]

    
    # feature1 = self.Flat(feature1)  
    # feature2 = self.Flat(feature2)
    # feature1 = Feature[0]
    # feature2 = Feature[1]
    # print(feature1.shape,feature2.shape)


    # feature1 = feature1.view(size,1)
    # feature2 = feature2.view(size,1)
    feature1_T = feature1.transpose(1,2)
    feature2_T = feature2.transpose(1,2)
    Gram1 = torch.bmm(feature1,feature1_T)
    Gram2 = torch.bmm(feature2,feature2_T)
    GRAM_LOSS = (1/(4 * size * size * channel_size * channel_size)) * torch.sum((Gram1 - Gram2)**2)
    
    return GRAM_LOSS

# --- Conv networks ---
def Conv4():
    return ConvNet(4)
def Conv6():
    return ConvNet(6)
def Conv4NP():
    return ConvNetNopool(4)
def Conv6NP():
    return ConvNetNopool(6)

# --- ResNet networks ---
def ResNet10(flatten=True, leakyrelu=False):
    return ResNet(SimpleBlock, [1,1,1,1], [64,128,256,512], flatten, leakyrelu)
def ResNet18(flatten=True, leakyrelu=False):
    return ResNet(SimpleBlock, [2,2,2,2], [64,128,256,512], flatten, leakyrelu)
def ResNet34(flatten=True, leakyrelu=False):
    return ResNet(SimpleBlock, [3,4,6,3], [64,128,256,512], flatten, leakyrelu)

model_dict = dict(Conv4=Conv4, Conv6=Conv6, ResNet10 = ResNet10, ResNet18 = ResNet18, ResNet34 = ResNet34)