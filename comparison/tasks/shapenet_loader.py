import re
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = re.findall('(.*)/tasks', BASE_DIR)[0]
sys.path.append(BASE_DIR)

from torch.utils.data import DataLoader
from data.shapenet import PartNormalDataset


shapenet = PartNormalDataset(split='test', npoints=2048, normalize=False, out_d=None)
test_loader = DataLoader(shapenet, num_workers=6,
                            batch_size=32, shuffle=False, drop_last=False)

print(len(shapenet))                    # > 2874
print(len(shapenet.__getitem__(0)))     # > 4
print(shapenet.__getitem__(0)[0].shape) # > 2048,3
print(shapenet.__getitem__(0)[1].shape) # > (1,)
print(shapenet.__getitem__(0)[2].shape) # > (2048,)
print(shapenet.__getitem__(0)[3].shape) # > 2874


shapenet = PartNormalDataset(split='test', npoints=2048, normalize=False, out_d=16)
test_loader = DataLoader(shapenet, num_workers=6,
                            batch_size=32, shuffle=False, drop_last=False)

print(len(shapenet))                    # > 2874
print(len(shapenet.__getitem__(0)))     # > 5
print(shapenet.__getitem__(0)[0].shape) # > 2048,3
print(shapenet.__getitem__(0)[1].shape) # > 2048,16
print(shapenet.__getitem__(0)[2].shape) # > 1,
print(shapenet.__getitem__(0)[3].shape) # 2048,
print(shapenet.__getitem__(0)[5].shape) #
'''
2874
4
(2048, 3)
(1,)
(2048,)
(2048, 3)

2874
5
(2048, 3)
(2048, 16)
(1,)
(2048,)
'''