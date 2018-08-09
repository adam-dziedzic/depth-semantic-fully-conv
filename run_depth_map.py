import argparse
import sys
import os
from model import Model, ResidualBlock, UpProj_Block
import flow_transforms
from torchvision import transforms
from nyu_dataset_loader import ListDataset
import torch
from matplotlib import pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("-g", "--is_debug", default=False, type=bool,
                    help="is it the debug mode execution")
parser.add_argument("-l", "--log_file", default="deeplens.log",
                    help="The name of the log file.")
parser.add_argument("-b", "--batch_size", default=16, type=int,
                    help="the initial size of the batch (number of data points "
                         "for a single forward and batch passes")

input_rgb_images_dir = 'data/nyu_datasets_changed/input/'
target_depth_images_dir = 'data/nyu_datasets_changed/target_depths/'
target_labels_images_dir = 'data/nyu_datasets_changed/labels_38/'

if torch.cuda.is_available():
    print("CUDA is available")
    dtype = torch.cuda.FloatTensor
else:
    print("No CUDA detected")
    dtype = torch.FloatTensor

args = parser.parse_args(sys.argv[1:])
is_debug = args.is_debug
log_file = args.log_file
batch_size = args.batch_size

model = Model(ResidualBlock, UpProj_Block, batch_size)
model.type(dtype)

# resume_file = 'model_first.pth.tar'
resume_file = 'checkpoint.pth.tar'
checkpoint = torch.load(resume_file)
model.load_state_dict(checkpoint['state_dict'])

input_rgb_images_dir = 'data/deeplens/input/'
listing = os.listdir(input_rgb_images_dir)

data_dir = (
    input_rgb_images_dir, input_rgb_images_dir, input_rgb_images_dir)

input_transform = transforms.Compose(
    [flow_transforms.Scale(228), flow_transforms.ArrayToTensor()])

# Apply this transform on input, ground truth depth images and labeled images

co_transform = flow_transforms.Compose([
    flow_transforms.RandomCrop((480, 640))
])

dataset = ListDataset(
    data_dir=data_dir, listing=listing, input_transform=input_transform,
    target_depth_transform=None, target_labels_transform=None,
    co_transform=co_transform, file_suffix="jpg")

data_loader = torch.utils.data.DataLoader(
    dataset=dataset, batch_size=1, shuffle=False, drop_last=False)

counter = 0
for x, y, z in data_loader:
    counter += 1
    x_var = torch.Variable(x.type(dtype), volatile=True)
    z_var = torch.Variable(z.type(dtype), volatile=True)

    pred_depth, _ = model(x_var, z_var)

    input_rgb_image = x_var[0].data.permute(1, 2,
                                            0).cpu().numpy().astype(
        np.uint8)
    plt.imsave('result_linput_rgb_counter_{}.png'.format(counter),
               input_rgb_image)

    input_gt_depth_image = z_var[0].data.permute(1, 2,
                                                 0).cpu().numpy().astype(
        np.uint8)
    plt.imsave('result_input_gt_depth_counter_{}.png'.format(counter),
               input_gt_depth_image)
