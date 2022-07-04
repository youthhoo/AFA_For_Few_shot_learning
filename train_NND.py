import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3,2'

import numpy as np
import torch
from data.datamgr import SetDataManager
from options import parse_args
from methods.NewFTNet import NewFTNet
import time

# training iterations
def train(base_loader, val_loader, model, start_epoch, stop_epoch, params):
  max_acc = 0.
  total_it = 0.
  for epoch in range(start_epoch, stop_epoch):
    since = time.time()
    model.train()
    total_it = model.train_loop(epoch, base_loader, total_it)
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    # validate
    model.eval()
    with torch.no_grad():
      acc = model.test_loop(val_loader)

    if acc > max_acc:
      print("best model! save...")
      max_acc = acc
      outfile = os.path.join(params.checkpoint_dir, 'best_model.tar')
      torch.save({'epoch':epoch, 'state':model.model.state_dict()}, outfile)
    else:
      print('GG!! best accuracy {:f}'.format(max_acc))

    if ((epoch+1)%params.save_freq==0) or (epoch==stop_epoch-1):
      outfile = os.path.join(params.checkpoint_dir, '{:d}.tar'.format(epoch))
      torch.save({'epoch':epoch, 'state':model.model.state_dict()}, outfile)
    

# --- main function ---
if __name__=='__main__':
  # set numpy random seed
  np.random.seed(10)

  # parse argument
  params = parse_args()
  print('--- FWT training ---')
  print(params)

  # output and tensorboard dir
  params.checkpoint_dir = '%s/checkpoints/%s'%(params.save_dir, params.name)
  if not os.path.isdir(params.checkpoint_dir):
    os.makedirs(params.checkpoint_dir)

  # dataloader
  print('\n--- prepare dataloader ---')
  print('\ttrain with single seen domain {}'.format(params.dataset))
  print('\tval with single seen domain {}'.format(params.dataset))
  # base_file = os.path.join(params.data_dir, params.dataset, 'base.json')
  # val_file = os.path.join(params.data_dir, params.testset, 'val.json')
  if params.parts == 'None':
    datapath = '/anonymous'
    base_file = os.path.join(datapath, params.dataset, 'base.json')
    val_file = os.path.join(datapath, params.dataset, 'val.json')
  else:
    datapath = '/anonymous'
    base_file = os.path.join(datapath, params.dataset, params.parts, 'base.json')
    val_file = os.path.join(datapath, params.testset, params.parts, 'val.json')

  # model
  image_size = 224
  n_query = max(1, int(16*params.test_n_way/params.train_n_way))
  base_datamgr = SetDataManager(image_size, n_query=n_query, n_way=params.train_n_way, n_support=params.n_shot)
  base_loader = base_datamgr.get_data_loader(base_file, aug=params.train_aug)
  val_datamgr = SetDataManager(image_size, n_query=n_query, n_way=params.test_n_way, n_support=params.n_shot)
  val_loader = val_datamgr.get_data_loader(val_file, aug=False)
  model = NewFTNet(params).cuda()

  # resume training
  start_epoch = params.start_epoch
  stop_epoch = params.stop_epoch
  if params.resume_epoch > 0:
      resume_file = os.path.join(params.checkpoint_dir, '{:d}.tar'.format(params.resume_epoch))
      tmp = torch.load(resume_file)
      start_epoch = tmp['epoch'] + 1
      model.model.load_state_dict(tmp['state'])
      print('\tResume the training weight at {} epoch.'.format(start_epoch))
  else:
      if params.parts == 'None':
        path = '%s/checkpoints/%s/399.tar' % (params.save_dir, params.resume_dir)
      else:
        path = '%s/checkpoints/pretrain/399.tar' % (params.save_dir)
      state = torch.load(path)['state']
      model_params = model.model.state_dict()
      pretrained_dict = {k: v for k, v in state.items() if k in model_params}
      print(pretrained_dict.keys())
      model_params.update(pretrained_dict)
      model.model.load_state_dict(model_params)
  
  net = torch.nn.DataParallel(model, device_ids = [0,1]).cuda()

  # training
  print('\n--- start the training ---')
  train(base_loader, val_loader, model, start_epoch, stop_epoch, params)