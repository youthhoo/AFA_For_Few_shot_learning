import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from methods import backbone
from methods.backbone import model_dict
from methods.matchingnet import MatchingNet
from methods.protonet import ProtoNet
from methods import relationnet
from methods import gnn
from methods import gnnnet
from methods import tpn

def Max_phase(model, ft_params,X_n, params):
    optimizer = optim.SGD(ft_params, lr=params.max_lr)
    model.eval()
    p = np.random.rand()
    if p > params.prob:
      for _ in range(params.T_max):
          optimizer.zero_grad()
          _, class_loss = model.set_forward_loss(X_n)
          (-class_loss).backward()
          optimizer.step()
    return ft_params

class NewFTNet(nn.Module):
  def __init__(self, params):
    super(NewFTNet, self).__init__()
    backbone.FeatureWiseTransformation2d_fw.feature_augment = True
    backbone.ConvBlock.FWT = True
    backbone.SimpleBlock.FWT = True
    backbone.ResNet.FWT = True
    self.params = params

    if params.method == 'ProtoNet':
      model = ProtoNet(model_dict[params.model], n_way=params.train_n_way, n_support=params.n_shot)
    elif params.method == 'MatchingNet':
      backbone.LSTMCell.FWT = True
      model = MatchingNet(model_dict[params.model], n_way=params.train_n_way, n_support=params.n_shot)
    elif params.method == 'RelationNet':
      relationnet.RelationConvBlock.FWT = True
      relationnet.RelationModule.FWT = True
      model = relationnet.RelationNet(model_dict[params.model], n_way=params.train_n_way, n_support=params.n_shot)
    elif params.method == 'GNN':
      gnnnet.GnnNet.FWT=True
      gnn.Gconv.FWT=True
      gnn.Wcompute.FWT=True
      model = gnnnet.GnnNet(model_dict[params.model], n_way=params.train_n_way, n_support=params.n_shot)
    elif params.method == 'TPN':
        tpn.RelationNetwork.FWT = True
        model = tpn.TPN(model_dict[params.model], n_way=params.train_n_way, n_support=params.n_shot).cuda()
    else:
      raise ValueError('Unknown method')
    self.model = model
    print('\ttrain with {} framework'.format(params.method))

    

    # total epochs
    self.total_epoch = params.stop_epoch

  def split_model_parameters(self):
    model_params = []
    ft_params = []
    for n, p in self.model.named_parameters():
      n = n.split('.')
      if n[-1] == 'gamma' or n[-1] == 'beta':
        ft_params.append(p)
        continue
        
      model_params.append(p)
    return model_params, ft_params

  def train_loop(self, epoch, base_loader, total_it):
    self.model.train()

    # optimizer
    model_params, ft_params = self.split_model_parameters()
    self.model_optim = torch.optim.Adam(model_params)

    ft_tmp = []
    # for idx in range(len(ft_params)):
    #     ft_tmp.append(ft_params[idx].detach())
    for weight in self.model.parameters():
      weight.fast = None

    # trainin loop
    print_freq = len(base_loader)//10
    avg_model_loss = 0.
    for i, (x, _) in enumerate(base_loader):
      self.model.n_query = x.size(1) - self.model.n_support
      if i > 150:
        tmp = Max_phase(self.model, ft_params, x, self.params)
    #   print(tmp == ft_params)
    #   for i in range(len(ft_params)):
    #       ft_params[i].copy_(tmp[i])
      _, model_loss = self.model.set_forward_loss(x)

      # optimize
      self.model_optim.zero_grad()
      model_loss.backward()
      self.model_optim.step()

      # loss
      avg_model_loss += model_loss.item()
      if (i+1)%print_freq==0:
        print('Epoch {:d}/{:d} | Batch {:d}/{:d} | loss {:f}'.format(epoch+1, self.total_epoch, i+1, len(base_loader), avg_model_loss/float(i+1)))
      total_it += 1
    return total_it

  def test_loop(self, test_loader, record=None):
    self.model.eval()
    for weight in self.model.parameters():
      weight.fast = None
    return self.model.test_loop(test_loader, record)

  def cuda(self):
    self.model.cuda()
    return self