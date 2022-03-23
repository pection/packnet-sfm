import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import numpy as np
import torch
import os
import sys

sys.path.append(os.getcwd())

from packnet_sfm.models.SelfSupModel import SelfSupModel

model = SelfSupModel()

PATH = '/home/ai/work/data/experiments/default_config-train_kitti-2022.03.14-09h57m43s/epoch=49_KITTI_raw-eigen_val_files-velodyne-abs_rel_pp_gt=0.089.ckpt'

 # model = Net()
# optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

checkpoint = torch.load(PATH)
print(checkpoint)
print(type(checkpoint))
# model.load_state_dict(checkpoint['model_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
# loss = checkpoint['loss']

model.eval()
output = model()
# print(epoch,loss)
#
# model.eval()
# # - or -
# model.train()
