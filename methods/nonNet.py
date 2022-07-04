import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from methods import backbone
from methods.backbone import model_dict
from methods.matchingnet import MatchingNet
from methods.protonet import ProtoNet
from methods import relationnet
from methods import gnn
from methods import gnnnet
from methods import tpn

def RCNN(X_n, params):  # (5, 21, 3, 224, 224)
    N, S, C, H, W = X_n.size()
    p = np.random.rand()
    K = [1, 3, 5, 7, 11, 15]
    if p > params.prob:
        k = K[np.random.randint(0, len(K))]
        Conv = nn.Conv2d(3, 3, kernel_size=k, stride=1, padding=k//2, bias=False)
        nn.init.xavier_normal_(Conv.weight)
        X_n = Conv(X_n.reshape(-1, C, H, W)).reshape(N, S, C, H, W)
    return X_n.detach()

def Max_phase(model, X_n, params):
    X_n = X_n.cuda()
    optimizer = optim.SGD([X_n.requires_grad_()], lr=params.max_lr)
    # print('X_N',type(X_n))
    # .requires_grad_())
    model.eval()
    for _ in range(params.T_max):
        optimizer.zero_grad()
        _, class_loss = model.set_forward_loss(X_n)
        (-class_loss).backward()
        optimizer.step()
    return X_n.detach()



class nonNet(nn.Module):
  def __init__(self, params):
    super(nonNet, self).__init__()
    backbone.FeatureWiseTransformation2d_fw.GRAM = True
    backbone.ConvBlock.GRAM = True
    backbone.SimpleBlock.GRAM = True
    backbone.ResNet.GRAM = True
    backbone.FeatureWiseTransformation2d_fw.NON = True
    self.params = params

    if params.method == 'ProtoNet':
      model = ProtoNet(model_dict[params.model], n_way=params.train_n_way, n_support=params.n_shot)
    elif params.method == 'MatchingNet':
    #   backbone.LSTMCell.FWT = True
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

    # optimizer
    model_params, Domain_params, Domain_ft_params = self.model_parameters()
    self.model_optim = torch.optim.Adam(model_params)
    self.Domain_optim = torch.optim.Adam(Domain_params)
    self.Domain_ft_optim = torch.optim.Adam(Domain_ft_params)

    # total epochs
    self.total_epoch = params.stop_epoch

  def model_parameters(self):
    model_params = []
    Domain_params = []
    Domain_ft_params = []
    for n, p in self.model.named_parameters():
        tmp = n.split('.')
        # if '7.C1_ND.weight' in n or '7.C1.weight' in n:
        #     print(p)
        if tmp[-1] == 'gamma' or tmp[-1] == 'beta' or tmp[-2] == 'Conv' or tmp[-2] == 'Conv2':
            Domain_ft_params.append(p)
        
        elif 'Domain' in n:
            Domain_params.append(p)
            Domain_ft_params.append(p)
        elif 'feature' in n:
            Domain_params.append(p)
            model_params.append(p)
            Domain_ft_params.append(p)
        else:
            model_params.append(p)
    return model_params, Domain_params, Domain_ft_params

  def train_loop(self, epoch, base_loader, total_it):
    self.model.train()
    for weight in self.model.parameters():
      weight.fast = None

    # trainin loop
    print_freq = len(base_loader)//10
    avg_model_loss = 0.
    for i, (x, _) in enumerate(base_loader):
      # if params != None:
      self.model.n_query = x.size(1) - self.model.n_support 
      # x = RCNN(x, self.params)
      # x = Max_phase(self.model, x, self.params)  # (5, 21, 3, 224, 224)
      self.model.train()
      
      _, model_loss, DC_loss = self.model.set_forward_loss(x, ND = True, epoch = epoch)

      # optimize
      self.model_optim.zero_grad()
      model_loss.backward(retain_graph = True)
      # eta = -10 
      # lamb = 2/(1 + math.exp(eta * (epoch / 400) ) ) - 1
      # print(DC_loss)
      # DC_loss
      if DC_loss != 0:
        DC_loss.backward()
    #   self.Domain_optim.zero_grad()
      # DC_loss.backward()
        self.model_optim.step()
        if epoch < 150:
          self.Domain_optim.step()
        else:
          self.Domain_ft_optim.step()
      else:
        self.model_optim.step()

      model_loss += DC_loss
      

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