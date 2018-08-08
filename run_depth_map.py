import argparse
import sys

from model import *

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

resume_file = 'model_best.pth.tar'
checkpoint = torch.load(resume_file)
model.load_state_dict(checkpoint['state_dict'])

