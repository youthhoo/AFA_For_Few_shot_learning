import sys
import os
from subprocess import call

if len(sys.argv) != 2:
    raise Exception('Incorrect command! e.g., python process.py DATASET [cars, cub, places, miniImagenet, plantae]')
dataset = sys.argv[1]

print('--- process ' + dataset + ' dataset ---')
if not os.path.exists(os.path.join(dataset, 'source')):
    os.makedirs(os.path.join(dataset, 'source'))
os.chdir(os.path.join(dataset, 'source'))

# download files
if dataset == 'cars':
    call('wget /anonymous', shell=True)
    call('tar -zxf /anonymous', shell=True)
    call('wget /anonymous', shell=True)
    call('tar -zxf /anonymous', shell=True)
elif dataset == 'cub':
    call('wget /anonymous', shell=True)
    call('tar -zxf /anonymous', shell=True)
elif dataset == 'places':
    call('wget /anonymous', shell=True)
    call('tar -xf /anonymous', shell=True)
# Since our experiments are based on Hung-Yu Tseng's implementation, we directly use the miniImagenet and plantae datasets from his project
elif dataset == 'miniImagenet':
    call('wget /anonymous', shell=True)
    call('tar -xjf /anonymous', shell=True)
elif dataset == 'plantae':
    call('wget /anonymous', shell=True)
    call('tar -xzf /anonymous', shell=True)
else:
    raise Exception('No such dataset!')

# process file
os.chdir('..')
call('python write_' + dataset + '_filelist.py', shell=True)
